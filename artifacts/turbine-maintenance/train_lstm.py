from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

from src.cmapss_data import (
    FEATURE_COLUMNS,
    NASA_CATALOG_URL,
    RUL_CAP_CYCLES,
    WINDOW_SIZE,
    _read_training_rows,
    ensure_training_file,
    fit_feature_scaler,
    make_engine_examples,
    make_windows,
    prepare_engine_split,
)
from src.lstm_model import NumpyLSTMRegressor

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = ROOT / "data" / "raw" / "train_FD001.txt"
DEFAULT_MODEL_DIR = ROOT / "model_artifacts"
SEED = 42
HIDDEN_SIZE = 24
BATCH_SIZE = 128
LEARNING_RATE = 0.002
MAX_EPOCHS = 24
EARLY_STOPPING_PATIENCE = 5
WINDOW_STRIDE = 5


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _write_predictions_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    columns = (
        "engine_id",
        "cycle",
        "actual_rul_cycles",
        "predicted_rul_cycles",
        "absolute_error_cycles",
    )
    with temporary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _update_status(model_dir: Path, payload: dict) -> None:
    _write_json_atomic(model_dir / "training_status.json", payload)


def train(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    model_dir: Path = DEFAULT_MODEL_DIR,
    epochs: int = MAX_EPOCHS,
    seed: int = SEED,
) -> dict:
    if epochs < 1:
        raise ValueError("epochs must be at least one")

    model_dir.mkdir(parents=True, exist_ok=True)
    _update_status(
        model_dir,
        {
            "status": "training",
            "message": "Training a NumPy LSTM on NASA C-MAPSS FD001 engine sequences.",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    try:
        local_dataset_path, data_source = ensure_training_file(dataset_path)
        rows = _read_training_rows(local_dataset_path)
        (
            training_rows,
            held_out_rows,
            training_engine_ids,
            held_out_engine_ids,
        ) = prepare_engine_split(rows, seed=seed)

        scaler = fit_feature_scaler(training_rows)
        means = np.asarray(scaler["means"], dtype=np.float32)
        scales = np.asarray(scaler["scales"], dtype=np.float32)

        training_windows, training_targets, _ = make_windows(
            training_rows,
            means,
            scales,
            stride=WINDOW_STRIDE,
        )
        held_out_windows, held_out_targets, held_out_window_engines = make_windows(
            held_out_rows,
            means,
            scales,
            stride=WINDOW_STRIDE,
        )

        model = NumpyLSTMRegressor(
            input_size=len(FEATURE_COLUMNS),
            hidden_size=HIDDEN_SIZE,
            seed=seed,
        )
        batch_rng = np.random.default_rng(seed)
        best_rmse = float("inf")
        best_epoch = 0
        best_parameters: dict[str, np.ndarray] = {}
        best_train_loss = float("inf")
        no_improvement_epochs = 0
        epochs_run = 0
        training_losses: list[float] = []
        validation_rmses: list[float] = []

        for epoch in range(1, epochs + 1):
            order = batch_rng.permutation(len(training_windows))
            batch_losses: list[float] = []
            for start in range(0, len(order), BATCH_SIZE):
                batch_indices = order[start : start + BATCH_SIZE]
                loss = model.train_batch(
                    training_windows[batch_indices],
                    training_targets[batch_indices],
                    learning_rate=LEARNING_RATE,
                )
                batch_losses.append(loss)

            epoch_loss = float(np.mean(batch_losses))
            held_out_predictions = np.clip(
                model.predict(held_out_windows),
                0.0,
                1.0,
            ) * RUL_CAP_CYCLES
            held_out_truth = held_out_targets * RUL_CAP_CYCLES
            validation_rmse = float(
                np.sqrt(np.mean((held_out_predictions - held_out_truth) ** 2))
            )
            if not np.isfinite(epoch_loss) or not np.isfinite(validation_rmse):
                raise FloatingPointError(
                    "LSTM training produced non-finite loss or evaluation values"
                )

            epochs_run = epoch
            training_losses.append(epoch_loss)
            validation_rmses.append(validation_rmse)
            if validation_rmse < best_rmse:
                best_rmse = validation_rmse
                best_epoch = epoch
                best_train_loss = epoch_loss
                best_parameters = {
                    name: parameter.copy()
                    for name, parameter in model._parameters().items()
                }
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1

            print(
                f"epoch={epoch:02d} "
                f"train_mse={epoch_loss:.6f} "
                f"held_out_rmse_cycles={validation_rmse:.3f}",
                flush=True,
            )
            if no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
                break

        for name, parameter in model._parameters().items():
            parameter[...] = best_parameters[name]

        final_predictions = np.clip(
            model.predict(held_out_windows),
            0.0,
            1.0,
        ) * RUL_CAP_CYCLES
        final_truth = held_out_targets * RUL_CAP_CYCLES
        final_rmse = float(np.sqrt(np.mean((final_predictions - final_truth) ** 2)))
        final_mae = float(np.mean(np.abs(final_predictions - final_truth)))

        examples = make_engine_examples(
            held_out_rows,
            means,
            scales,
            seed=seed + 1,
        )
        if not examples:
            raise ValueError(
                "Could not create an example prediction for any held-out engine"
            )
        example_windows = np.asarray(
            [example["window"] for example in examples],
            dtype=np.float32,
        )
        example_predictions = np.clip(
            model.predict(example_windows),
            0.0,
            1.0,
        ) * RUL_CAP_CYCLES

        prediction_rows: list[dict] = []
        for example, prediction in zip(examples, example_predictions):
            actual = int(example["actual_rul_cycles"])
            predicted = float(prediction)
            prediction_rows.append(
                {
                    "engine_id": int(example["engine_id"]),
                    "cycle": int(example["cycle"]),
                    "actual_rul_cycles": actual,
                    "predicted_rul_cycles": round(predicted, 2),
                    "absolute_error_cycles": round(abs(predicted - actual), 2),
                }
            )

        model_path = model_dir / "fd001_lstm.npz"
        preprocessing_path = model_dir / "preprocessing.json"
        predictions_path = model_dir / "heldout_engine_predictions.csv"
        demo_sequences_path = model_dir / "heldout_demo_sequences.npz"
        report_path = model_dir / "training_report.json"

        model.save(model_path)
        demo_sequences_tmp = demo_sequences_path.with_suffix(".npz.tmp")
        with demo_sequences_tmp.open("wb") as output:
            np.savez_compressed(
                output,
                normalized_windows=example_windows,
                engine_ids=np.asarray(
                    [example["engine_id"] for example in examples],
                    dtype=np.int64,
                ),
                cycles=np.asarray(
                    [example["cycle"] for example in examples],
                    dtype=np.int64,
                ),
                actual_rul_cycles=np.asarray(
                    [example["actual_rul_cycles"] for example in examples],
                    dtype=np.int64,
                ),
            )
        demo_sequences_tmp.replace(demo_sequences_path)
        preprocessing = {
            **scaler,
            "window_size": WINDOW_SIZE,
            "rul_cap_cycles": RUL_CAP_CYCLES,
            "input_size": len(FEATURE_COLUMNS),
            "hidden_size": HIDDEN_SIZE,
            "target_scale": RUL_CAP_CYCLES,
            "normalization": "z-score; fitted only on training engines",
            "window_order": "oldest cycle to newest cycle",
        }
        _write_json_atomic(preprocessing_path, preprocessing)
        _write_predictions_csv(predictions_path, prediction_rows)

        report = {
            "status": "trained",
            "model_name": "Single-layer NumPy LSTM regressor",
            "dataset": "NASA C-MAPSS FD001 turbofan training sequences",
            "data_source": data_source,
            "dataset_catalog": NASA_CATALOG_URL,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "engine_split": "Group split by engine ID before preprocessing and window creation",
            "training_engine_count": len(training_engine_ids),
            "held_out_engine_count": len(held_out_engine_ids),
            "training_engine_ids": training_engine_ids,
            "held_out_engine_ids": held_out_engine_ids,
            "training_window_count": int(len(training_windows)),
            "held_out_window_count": int(len(held_out_windows)),
            "held_out_rmse_cycles": round(final_rmse, 4),
            "held_out_mae_cycles": round(final_mae, 4),
            "held_out_metric_target": (
                "All held-out engine windows; RUL target capped at 125 cycles"
            ),
            "example_prediction_count": len(prediction_rows),
            "training_config": {
                "seed": seed,
                "window_size": WINDOW_SIZE,
                "window_stride": WINDOW_STRIDE,
                "rul_cap_cycles": RUL_CAP_CYCLES,
                "hidden_size": HIDDEN_SIZE,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "max_epochs": epochs,
                "epochs_run": epochs_run,
                "best_epoch": best_epoch,
                "best_epoch_training_mse": round(best_train_loss, 7),
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "features": FEATURE_COLUMNS,
                "optimizer": "Adam",
                "training_loss_by_epoch": [
                    round(value, 7) for value in training_losses
                ],
                "held_out_rmse_by_epoch_cycles": [
                    round(value, 4) for value in validation_rmses
                ],
            },
            "artifact_files": {
                "model": model_path.name,
                "preprocessing": preprocessing_path.name,
                "held_out_predictions": predictions_path.name,
                "demo_sequences": demo_sequences_path.name,
            },
            "important_scope_note": (
                "This is a C-MAPSS turbofan benchmark. Its 24 features and units "
                "do not match the four simulated TURBINE_01 sensors. The model "
                "is displayed separately and does not drive TURBINE_01 alerts."
            ),
        }
        _write_json_atomic(report_path, report)
        _update_status(model_dir, report)
        return report
    except Exception as error:
        failure = {
            "status": "failed",
            "message": str(error),
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(dataset_path),
            "traceback": traceback.format_exc(limit=8),
        }
        _update_status(model_dir, failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an LSTM on NASA C-MAPSS FD001."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to train_FD001.txt; downloads from NASA or a public mirror if absent.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory for model weights, preprocessing configuration, and metrics.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=MAX_EPOCHS,
        help="Maximum training epochs (early stopping may stop sooner).",
    )
    args = parser.parse_args()

    try:
        report = train(
            dataset_path=args.dataset,
            model_dir=args.model_dir,
            epochs=args.epochs,
        )
    except Exception as error:
        print(f"LSTM training failed: {error}", file=sys.stderr)
        return 1

    print(
        "Training complete. Held-out engine RMSE: "
        f"{report['held_out_rmse_cycles']:.4f} cycles "
        f"({report['held_out_engine_count']} engines)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
