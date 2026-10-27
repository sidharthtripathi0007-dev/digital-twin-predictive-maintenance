from __future__ import annotations

import io
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

NASA_CATALOG_URL = "https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data"
NASA_ARCHIVE_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
PUBLIC_MIRROR_URL = (
    "https://raw.githubusercontent.com/cyrilli/TurboEngine_Dataset_NASA/"
    "master/CMAPSSData/train_FD001.txt"
)

RAW_COLUMNS = (
    ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{index}" for index in range(1, 22)]
)
FEATURE_COLUMNS = RAW_COLUMNS[2:]
WINDOW_SIZE = 30
RUL_CAP_CYCLES = 125


def ensure_training_file(dataset_path: Path) -> tuple[Path, str]:
    """Find a cached NASA training file or retrieve it from a public source."""
    if dataset_path.exists():
        _read_training_rows(dataset_path)
        return dataset_path, "Local NASA C-MAPSS FD001 training file"

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    try:
        request = urllib.request.Request(
            NASA_ARCHIVE_URL,
            headers={"User-Agent": "TURBINE-01-College-Project/1.0"},
        )
        with urllib.request.urlopen(request, timeout=25) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            member = next(
                name
                for name in archive.namelist()
                if name.lower().endswith("train_fd001.txt")
            )
            content = archive.read(member)
        dataset_path.write_bytes(content)
        _read_training_rows(dataset_path)
        return dataset_path, NASA_CATALOG_URL
    except Exception as error:
        errors.append(f"NASA archive: {error}")

    try:
        request = urllib.request.Request(
            PUBLIC_MIRROR_URL,
            headers={"User-Agent": "TURBINE-01-College-Project/1.0"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()
        dataset_path.write_bytes(content)
        _read_training_rows(dataset_path)
        return dataset_path, f"{NASA_CATALOG_URL} (public mirror: GitHub)"
    except Exception as error:
        errors.append(f"Public mirror: {error}")

    dataset_path.unlink(missing_ok=True)
    raise RuntimeError(
        "Could not obtain NASA C-MAPSS FD001 training data. "
        + " | ".join(errors)
    )


def _read_training_rows(dataset_path: Path) -> np.ndarray:
    try:
        rows = np.loadtxt(dataset_path, dtype=np.float32, ndmin=2)
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not parse FD001 training file: {error}") from error

    if rows.ndim != 2 or rows.shape[1] != len(RAW_COLUMNS):
        raise ValueError(
            f"Expected 26 numeric columns in train_FD001.txt; got shape {rows.shape}"
        )
    if not len(rows) or not np.isfinite(rows).all():
        raise ValueError("FD001 training data is empty or contains non-finite values")

    engine_ids = rows[:, 0].astype(np.int64)
    if np.any(engine_ids < 1) or len(np.unique(engine_ids)) < 2:
        raise ValueError("FD001 training file must contain multiple engine sequences")
    return rows


def prepare_engine_split(
    rows: np.ndarray,
    seed: int = 42,
    validation_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Split by engine before fitting preprocessing or making windows."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")

    engine_ids = np.unique(rows[:, 0].astype(np.int64))
    if len(engine_ids) < 5:
        raise ValueError("At least five distinct engines are required for a split")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(engine_ids)
    validation_count = max(1, int(round(len(engine_ids) * validation_fraction)))
    validation_ids = sorted(int(value) for value in shuffled[:validation_count])
    training_ids = sorted(int(value) for value in shuffled[validation_count:])
    training_rows = rows[np.isin(rows[:, 0].astype(np.int64), training_ids)]
    validation_rows = rows[np.isin(rows[:, 0].astype(np.int64), validation_ids)]
    return training_rows, validation_rows, training_ids, validation_ids


def fit_feature_scaler(training_rows: np.ndarray) -> dict[str, list[float]]:
    feature_values = training_rows[:, 2:].astype(np.float64)
    means = feature_values.mean(axis=0)
    scales = feature_values.std(axis=0)
    scales[scales < 1e-8] = 1.0
    return {
        "feature_columns": FEATURE_COLUMNS,
        "means": means.astype(float).tolist(),
        "scales": scales.astype(float).tolist(),
    }


def make_windows(
    rows: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    window_size: int = WINDOW_SIZE,
    rul_cap_cycles: int = RUL_CAP_CYCLES,
    stride: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build chronological sliding windows while preserving engine boundaries."""
    if window_size < 2 or stride < 1:
        raise ValueError("window_size must be >= 2 and stride must be positive")

    windows: list[np.ndarray] = []
    targets: list[float] = []
    engine_for_window: list[int] = []

    engine_ids = np.unique(rows[:, 0].astype(np.int64))
    for engine_id in engine_ids:
        sequence = rows[rows[:, 0].astype(np.int64) == engine_id]
        sequence = sequence[np.argsort(sequence[:, 1])]
        if len(sequence) < window_size:
            continue

        raw_features = sequence[:, 2:].astype(np.float32)
        normalized = (raw_features - means) / scales
        remaining_cycles = np.maximum(
            sequence[-1, 1] - sequence[:, 1],
            0.0,
        )
        target_values = np.minimum(remaining_cycles, rul_cap_cycles)

        end_indices = list(range(window_size - 1, len(sequence), stride))
        if end_indices[-1] != len(sequence) - 1:
            end_indices.append(len(sequence) - 1)
        for end_index in end_indices:
            start_index = end_index - window_size + 1
            windows.append(normalized[start_index : end_index + 1])
            targets.append(float(target_values[end_index]) / rul_cap_cycles)
            engine_for_window.append(int(engine_id))

    if not windows:
        raise ValueError("No engine sequences are long enough for LSTM windows")

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(engine_for_window, dtype=np.int64),
    )


def make_engine_examples(
    rows: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    seed: int = 123,
    window_size: int = WINDOW_SIZE,
    rul_cap_cycles: int = RUL_CAP_CYCLES,
) -> list[dict[str, float | int]]:
    """Choose one reproducible pre-failure window per held-out engine."""
    rng = np.random.default_rng(seed)
    examples: list[dict[str, float | int]] = []
    engine_ids = np.unique(rows[:, 0].astype(np.int64))

    for engine_id in engine_ids:
        sequence = rows[rows[:, 0].astype(np.int64) == engine_id]
        sequence = sequence[np.argsort(sequence[:, 1])]
        last_cycle = float(sequence[-1, 1])
        remaining = last_cycle - sequence[:, 1]
        eligible = np.flatnonzero(
            (np.arange(len(sequence)) >= window_size - 1)
            & (remaining >= 20.0)
            & (remaining <= rul_cap_cycles)
        )
        if not len(eligible):
            continue

        end_index = int(rng.choice(eligible))
        start_index = end_index - window_size + 1
        normalized = (
            sequence[start_index : end_index + 1, 2:].astype(np.float32) - means
        ) / scales
        examples.append(
            {
                "engine_id": int(engine_id),
                "cycle": int(sequence[end_index, 1]),
                "actual_rul_cycles": int(remaining[end_index]),
                "window": normalized,
            }
        )
    return examples
