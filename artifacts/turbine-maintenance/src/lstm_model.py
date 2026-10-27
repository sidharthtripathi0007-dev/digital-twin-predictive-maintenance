from __future__ import annotations

from pathlib import Path

import numpy as np


class NumpyLSTMRegressor:
    """A small, trainable one-layer LSTM regressor implemented with NumPy.

    Gates use the standard input/forget/candidate/output recurrence. Training
    uses batched backpropagation through time and Adam; no mock predictions or
    framework-specific checkpoint format are involved.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 24,
        seed: int = 42,
    ) -> None:
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be positive")

        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (input_size + 4 * hidden_size))
        self.Wx = rng.uniform(
            -limit,
            limit,
            size=(input_size, 4 * hidden_size),
        ).astype(np.float32)
        self.Wh = rng.normal(
            0.0,
            1.0 / np.sqrt(hidden_size),
            size=(hidden_size, 4 * hidden_size),
        ).astype(np.float32)
        self.bias = np.zeros(4 * hidden_size, dtype=np.float32)
        self.bias[hidden_size : 2 * hidden_size] = 1.0
        self.Wy = rng.normal(
            0.0,
            1.0 / np.sqrt(hidden_size),
            size=(hidden_size, 1),
        ).astype(np.float32)
        self.by = np.zeros(1, dtype=np.float32)

        self._step = 0
        self._first_moments = {
            name: np.zeros_like(value) for name, value in self._parameters().items()
        }
        self._second_moments = {
            name: np.zeros_like(value) for name, value in self._parameters().items()
        }

    def _parameters(self) -> dict[str, np.ndarray]:
        return {
            "Wx": self.Wx,
            "Wh": self.Wh,
            "bias": self.bias,
            "Wy": self.Wy,
            "by": self.by,
        }

    def _forward(
        self,
        windows: np.ndarray,
        keep_cache: bool,
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, ...]], np.ndarray]:
        windows = np.asarray(windows, dtype=np.float32)
        if windows.ndim != 3 or windows.shape[2] != self.input_size:
            raise ValueError(
                "Expected windows with shape "
                f"(batch, time, {self.input_size}); got {windows.shape}"
            )

        batch_size, sequence_length, _ = windows.shape
        hidden_size = self.hidden_size
        hidden = np.zeros((batch_size, hidden_size), dtype=np.float32)
        cell = np.zeros_like(hidden)
        caches: list[tuple[np.ndarray, ...]] = []

        for time_index in range(sequence_length):
            current_input = windows[:, time_index, :]
            previous_hidden = hidden
            previous_cell = cell
            gates = (
                current_input @ self.Wx
                + previous_hidden @ self.Wh
                + self.bias
            )

            input_gate = self._sigmoid(gates[:, :hidden_size])
            forget_gate = self._sigmoid(
                gates[:, hidden_size : 2 * hidden_size]
            )
            candidate = np.tanh(gates[:, 2 * hidden_size : 3 * hidden_size])
            output_gate = self._sigmoid(gates[:, 3 * hidden_size :])

            cell = forget_gate * previous_cell + input_gate * candidate
            tanh_cell = np.tanh(cell)
            hidden = output_gate * tanh_cell

            if keep_cache:
                caches.append(
                    (
                        current_input,
                        previous_hidden,
                        previous_cell,
                        input_gate,
                        forget_gate,
                        candidate,
                        output_gate,
                        tanh_cell,
                    )
                )

        prediction = hidden @ self.Wy + self.by
        return prediction[:, 0], caches, hidden

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def predict(self, windows: np.ndarray) -> np.ndarray:
        predictions, _, _ = self._forward(windows, keep_cache=False)
        return predictions

    def train_batch(
        self,
        windows: np.ndarray,
        targets: np.ndarray,
        learning_rate: float = 0.002,
    ) -> float:
        targets = np.asarray(targets, dtype=np.float32).reshape(-1)
        predictions, caches, final_hidden = self._forward(
            windows,
            keep_cache=True,
        )
        if len(targets) != len(predictions):
            raise ValueError("The number of targets must match the batch size")

        errors = predictions - targets
        loss = float(np.mean(errors * errors))
        gradient_output = (2.0 / len(targets)) * errors[:, None]

        gradients = {
            "Wx": np.zeros_like(self.Wx),
            "Wh": np.zeros_like(self.Wh),
            "bias": np.zeros_like(self.bias),
            "Wy": final_hidden.T @ gradient_output,
            "by": np.sum(gradient_output, axis=0),
        }
        hidden_gradient = gradient_output @ self.Wy.T
        cell_gradient = np.zeros_like(hidden_gradient)
        hidden_size = self.hidden_size

        for (
            current_input,
            previous_hidden,
            previous_cell,
            input_gate,
            forget_gate,
            candidate,
            output_gate,
            tanh_cell,
        ) in reversed(caches):
            output_gradient = hidden_gradient * tanh_cell
            output_pre_gradient = (
                output_gradient * output_gate * (1.0 - output_gate)
            )

            total_cell_gradient = (
                cell_gradient
                + hidden_gradient
                * output_gate
                * (1.0 - tanh_cell * tanh_cell)
            )
            forget_pre_gradient = (
                total_cell_gradient
                * previous_cell
                * forget_gate
                * (1.0 - forget_gate)
            )
            input_pre_gradient = (
                total_cell_gradient
                * candidate
                * input_gate
                * (1.0 - input_gate)
            )
            candidate_pre_gradient = (
                total_cell_gradient
                * input_gate
                * (1.0 - candidate * candidate)
            )
            gate_gradient = np.concatenate(
                (
                    input_pre_gradient,
                    forget_pre_gradient,
                    candidate_pre_gradient,
                    output_pre_gradient,
                ),
                axis=1,
            )

            gradients["Wx"] += current_input.T @ gate_gradient
            gradients["Wh"] += previous_hidden.T @ gate_gradient
            gradients["bias"] += np.sum(gate_gradient, axis=0)
            hidden_gradient = gate_gradient @ self.Wh.T
            cell_gradient = total_cell_gradient * forget_gate

        gradient_norm = np.sqrt(
            sum(
                float(np.sum(gradient * gradient, dtype=np.float64))
                for gradient in gradients.values()
            )
        )
        if not np.isfinite(gradient_norm) or not np.isfinite(loss):
            raise FloatingPointError("Non-finite LSTM loss or gradient")
        clip_scale = min(1.0, 5.0 / max(gradient_norm, 1e-12))
        gradients = {
            name: gradient * clip_scale
            for name, gradient in gradients.items()
        }

        self._step += 1
        beta_one = 0.9
        beta_two = 0.999
        for name, parameter in self._parameters().items():
            gradient = gradients[name]
            first = self._first_moments[name]
            second = self._second_moments[name]
            first *= beta_one
            first += (1.0 - beta_one) * gradient
            second *= beta_two
            second += (1.0 - beta_two) * gradient * gradient

            corrected_first = first / (1.0 - beta_one**self._step)
            corrected_second = second / (1.0 - beta_two**self._step)
            parameter -= (
                learning_rate
                * corrected_first
                / (np.sqrt(corrected_second) + 1e-8)
            )

        return loss

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("wb") as output:
            np.savez_compressed(output, **self._parameters())
        temporary_path.replace(path)

    @classmethod
    def load(
        cls,
        path: Path,
        input_size: int,
        hidden_size: int,
    ) -> "NumpyLSTMRegressor":
        model = cls(input_size=input_size, hidden_size=hidden_size)
        with np.load(path, allow_pickle=False) as saved:
            for name, parameter in model._parameters().items():
                stored = saved[name]
                if stored.shape != parameter.shape:
                    raise ValueError(f"Saved LSTM parameter has wrong shape: {name}")
                parameter[...] = stored
        return model
