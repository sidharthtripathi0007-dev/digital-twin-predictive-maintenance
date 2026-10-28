from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def load_heldout_predictions(
    model_dir: Path,
    model,
    config: dict,
) -> tuple[dict, pd.DataFrame]:
    """Run the saved model on preprocessed, engine-disjoint FD001 examples."""
    report_path = model_dir / "training_report.json"
    demo_path = model_dir / "heldout_demo_sequences.npz"
    saved_predictions_path = model_dir / "heldout_engine_predictions.csv"

    if not report_path.exists():
        raise FileNotFoundError("The saved LSTM training report is missing.")
    if not demo_path.exists():
        raise FileNotFoundError(
            "The saved held-out FD001 inference sequences are missing."
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "trained":
        raise ValueError("The saved training report does not mark the model as trained.")

    expected_shape = (
        int(config["window_size"]),
        int(config["input_size"]),
    )
    with np.load(demo_path, allow_pickle=False) as saved:
        windows = np.asarray(saved["normalized_windows"], dtype=np.float32)
        engine_ids = np.asarray(saved["engine_ids"], dtype=np.int64)
        cycles = np.asarray(saved["cycles"], dtype=np.int64)
        actual_rul = np.asarray(saved["actual_rul_cycles"], dtype=np.int64)

    if windows.ndim != 3 or tuple(windows.shape[1:]) != expected_shape:
        raise ValueError(
            "The saved demo windows do not match the model configuration: "
            f"expected (samples, {expected_shape[0]}, {expected_shape[1]}), "
            f"received {windows.shape}."
        )
    if not (len(windows) == len(engine_ids) == len(cycles) == len(actual_rul)):
        raise ValueError("The held-out demo sequence metadata is inconsistent.")
    if not len(windows) or not np.isfinite(windows).all():
        raise ValueError("The held-out demo sequences are empty or non-finite.")

    target_scale = float(config["target_scale"])
    rul_cap = float(config["rul_cap_cycles"])
    if target_scale <= 0 or rul_cap <= 0:
        raise ValueError("The saved RUL scaling configuration is invalid.")

    prediction_values = np.clip(
        model.predict(windows),
        0.0,
        1.0,
    ) * target_scale
    prediction_values = np.clip(prediction_values, 0.0, rul_cap)
    predictions = pd.DataFrame(
        {
            "engine_id": engine_ids,
            "cycle": cycles,
            "actual_rul_cycles": actual_rul,
            "predicted_rul_cycles": prediction_values,
        }
    )
    predictions["absolute_error_cycles"] = np.abs(
        predictions["predicted_rul_cycles"] - predictions["actual_rul_cycles"]
    )

    if saved_predictions_path.exists():
        saved_predictions = pd.read_csv(saved_predictions_path).sort_values(
            "engine_id"
        )
        current_predictions = predictions.sort_values("engine_id")
        expected_columns = ("engine_id", "cycle", "actual_rul_cycles")
        if len(saved_predictions) != len(current_predictions):
            raise ValueError("Saved demo inputs do not match the evaluation CSV.")
        for column in expected_columns:
            if not np.array_equal(
                saved_predictions[column].to_numpy(),
                current_predictions[column].to_numpy(),
            ):
                raise ValueError(
                    f"Saved demo inputs do not match evaluation field {column}."
                )
        if not np.allclose(
            saved_predictions["predicted_rul_cycles"].to_numpy(dtype=float),
            current_predictions["predicted_rul_cycles"].to_numpy(dtype=float),
            atol=0.011,
            rtol=0.0,
        ):
            raise ValueError(
                "The loaded LSTM does not reproduce its saved held-out predictions."
            )

    return report, predictions.sort_values("engine_id").reset_index(drop=True)


def parse_cmapss_csv(
    content: bytes,
    config: dict,
) -> list[dict[str, str | np.ndarray]]:
    """Parse named-feature or raw 24/26-column C-MAPSS CSV/text uploads."""
    if not content:
        raise ValueError("The uploaded file is empty.")

    feature_columns = list(config["feature_columns"])
    input_size = int(config["input_size"])
    window_size = int(config["window_size"])
    if len(feature_columns) != input_size:
        raise ValueError(
            "Saved preprocessing feature count does not match the model input size."
        )

    frame = _read_named_csv(content)
    if frame is not None:
        normalized_columns = {
            _normalize_column_name(column): column for column in frame.columns
        }
        feature_map = {
            feature: normalized_columns[_normalize_column_name(feature)]
            for feature in feature_columns
            if _normalize_column_name(feature) in normalized_columns
        }
        if not feature_map:
            frame = None
        elif len(feature_map) != input_size:
            missing = [
                feature
                for feature in feature_columns
                if feature not in feature_map
            ]
            raise ValueError(
                f"This model requires {input_size} input features in its saved "
                f"order; the CSV is missing: {', '.join(missing)}. "
                "The current checkpoint uses 3 operating settings plus 21 sensors."
            )

    if frame is not None:
        values = pd.DataFrame(
            {
                feature: pd.to_numeric(frame[feature_map[feature]], errors="raise")
                for feature in feature_columns
            }
        )
        unit_column = _find_column(
            normalized_columns,
            {"unit", "unitid", "unitnumber", "engine", "engineid"},
        )
        cycle_column = _find_column(
            normalized_columns,
            {"cycle", "cycles", "time", "timeincycles"},
        )
        unit_values = frame[unit_column] if unit_column else None
        cycle_values = (
            pd.to_numeric(frame[cycle_column], errors="raise")
            if cycle_column
            else None
        )
    else:
        raw = _read_headerless_numeric(content)
        if raw.shape[1] == 26:
            unit_values = raw[:, 0]
            cycle_values = raw[:, 1]
            values = pd.DataFrame(raw[:, 2:], columns=feature_columns)
        elif raw.shape[1] == input_size:
            unit_values = None
            cycle_values = None
            values = pd.DataFrame(raw, columns=feature_columns)
        elif raw.shape[1] == 21:
            raise ValueError(
                "The saved checkpoint requires 24 input channels: 3 C-MAPSS "
                "operating settings and 21 sensors. Sensor-only CSVs cannot be "
                "used without those real settings."
            )
        else:
            raise ValueError(
                f"Expected a CSV with {input_size} feature columns or the "
                f"26-column raw C-MAPSS layout; found {raw.shape[1]} columns."
            )

    numeric_values = values.to_numpy(dtype=np.float32)
    if not len(numeric_values) or not np.isfinite(numeric_values).all():
        raise ValueError("The uploaded feature values are empty or non-finite.")

    if unit_values is None:
        groups = [("Uploaded sequence", np.arange(len(values)))]
    else:
        units = pd.Series(unit_values).reset_index(drop=True)
        groups = [
            (
                f"Engine {_format_engine_id(engine_id)}",
                np.flatnonzero(units.to_numpy() == engine_id),
            )
            for engine_id in pd.unique(units)
        ]

    sequences: list[dict[str, str | np.ndarray]] = []
    for label, row_indices in groups:
        ordered_indices = np.asarray(row_indices, dtype=np.int64)
        if cycle_values is not None:
            group_cycles = np.asarray(cycle_values)[ordered_indices]
            ordered_indices = ordered_indices[np.argsort(group_cycles)]
        if len(ordered_indices) < window_size:
            continue
        selected_indices = ordered_indices[-window_size:]
        if cycle_values is not None:
            selected_cycles = np.asarray(cycle_values)[selected_indices]
            if not np.allclose(np.diff(selected_cycles), 1.0, rtol=0.0, atol=1e-6):
                raise ValueError(
                    f"{label} does not contain {window_size} consecutive cycles."
                )
        sequences.append(
            {
                "label": label,
                "sequence": numeric_values[selected_indices],
            }
        )

    if not sequences:
        raise ValueError(
            f"No engine sequence has the required {window_size} consecutive cycles."
        )
    return sequences


def _read_named_csv(content: bytes) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except (pd.errors.ParserError, UnicodeDecodeError):
        return None
    if frame.empty:
        return None
    return frame


def _read_headerless_numeric(content: bytes) -> np.ndarray:
    candidates: list[pd.DataFrame] = []
    for separator, engine in ((",", "c"), (r"\s+", "python")):
        try:
            candidate = pd.read_csv(
                io.BytesIO(content),
                header=None,
                sep=separator,
                engine=engine,
            )
        except (pd.errors.ParserError, UnicodeDecodeError):
            continue
        if candidate.empty:
            continue
        try:
            numeric = candidate.apply(pd.to_numeric, errors="raise")
        except (ValueError, TypeError):
            continue
        if numeric.shape[1] in (21, 24, 26):
            candidates.append(numeric)
    if not candidates:
        raise ValueError(
            "Could not parse the upload. Use a CSV with the saved feature names "
            "or headerless C-MAPSS rows."
        )
    widest = max(candidates, key=lambda candidate: candidate.shape[1])
    return widest.to_numpy(dtype=np.float32)


def _normalize_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _find_column(
    normalized_columns: dict[str, object],
    aliases: set[str],
) -> object | None:
    for alias in aliases:
        if alias in normalized_columns:
            return normalized_columns[alias]
    return None


def _format_engine_id(engine_id: object) -> str:
    try:
        numeric_id = float(engine_id)
    except (TypeError, ValueError):
        return str(engine_id)
    if numeric_id.is_integer():
        return str(int(numeric_id))
    return str(engine_id)
