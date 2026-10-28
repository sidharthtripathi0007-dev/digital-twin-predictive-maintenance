import io
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.lstm_inference import load_heldout_predictions, parse_cmapss_csv
from src.lstm_model import NumpyLSTMRegressor


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "model_artifacts"


class LSTMInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (MODEL_DIR / "preprocessing.json").read_text(encoding="utf-8")
        )
        cls.model = NumpyLSTMRegressor.load(
            MODEL_DIR / "fd001_lstm.npz",
            input_size=cls.config["input_size"],
            hidden_size=cls.config["hidden_size"],
        )

    def test_heldout_demo_runs_the_saved_model_and_matches_saved_predictions(self):
        report, predictions = load_heldout_predictions(
            MODEL_DIR,
            self.model,
            self.config,
        )

        self.assertEqual(report["held_out_rmse_cycles"], 14.8656)
        self.assertEqual(report["held_out_mae_cycles"], 10.847)
        self.assertEqual(len(predictions), report["held_out_engine_count"])
        saved = pd.read_csv(MODEL_DIR / "heldout_engine_predictions.csv")
        np.testing.assert_array_equal(
            predictions["engine_id"].to_numpy(),
            saved.sort_values("engine_id")["engine_id"].to_numpy(),
        )
        np.testing.assert_allclose(
            predictions["predicted_rul_cycles"].to_numpy(),
            saved.sort_values("engine_id")["predicted_rul_cycles"].to_numpy(),
            atol=0.011,
            rtol=0.0,
        )

    def test_named_feature_csv_returns_the_latest_30_cycles(self):
        features = self.config["feature_columns"]
        values = np.arange(35 * len(features), dtype=np.float32).reshape(
            35,
            len(features),
        )
        frame = pd.DataFrame(values, columns=features)
        sequences = parse_cmapss_csv(
            frame.to_csv(index=False).encode("utf-8"),
            self.config,
        )

        self.assertEqual(len(sequences), 1)
        self.assertEqual(sequences[0]["sequence"].shape, (30, 24))
        np.testing.assert_array_equal(
            sequences[0]["sequence"],
            values[-30:],
        )

    def test_raw_headerless_cmapss_rows_are_grouped_and_cycle_sorted(self):
        features = len(self.config["feature_columns"])
        rows = []
        for engine_id in (7, 8):
            for cycle in range(1, 36):
                sensor_values = np.full(features, engine_id + cycle, dtype=float)
                rows.append([engine_id, cycle, *sensor_values])
        output = io.StringIO()
        np.savetxt(output, np.asarray(rows), delimiter=",")

        sequences = parse_cmapss_csv(
            output.getvalue().encode("utf-8"),
            self.config,
        )

        self.assertEqual(
            [item["label"] for item in sequences],
            ["Engine 7", "Engine 8"],
        )
        self.assertEqual(sequences[0]["sequence"].shape, (30, 24))
        self.assertEqual(float(sequences[0]["sequence"][-1, 0]), 42.0)

    def test_sensor_only_csv_is_rejected_without_inventing_settings(self):
        frame = pd.DataFrame(
            np.ones((30, 21)),
            columns=[f"sensor_{index}" for index in range(1, 22)],
        )

        with self.assertRaisesRegex(ValueError, "24 input features"):
            parse_cmapss_csv(
                frame.to_csv(index=False).encode("utf-8"),
                self.config,
            )

    def test_named_sequence_with_cycle_gaps_is_rejected(self):
        features = self.config["feature_columns"]
        cycles = [cycle for cycle in range(1, 37) if cycle != 19]
        frame = pd.DataFrame(
            {
                "unit_id": [7] * len(cycles),
                "cycle": cycles,
                **{
                    feature: np.ones(len(cycles), dtype=np.float32)
                    for feature in features
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "consecutive cycles"):
            parse_cmapss_csv(
                frame.to_csv(index=False).encode("utf-8"),
                self.config,
            )


if __name__ == "__main__":
    unittest.main()
