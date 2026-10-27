from __future__ import annotations

import random
from datetime import datetime

CONDITIONS = ("Healthy", "Degrading", "Critical")

_PROFILES = {
    "Healthy": {
        "temperature": (69.0, 2.0),
        "vibration": (0.65, 0.15),
        "pressure": (5.35, 0.10),
        "speed": (1800.0, 12.0),
        "health": (92.0, 99.0),
    },
    "Degrading": {
        "temperature": (83.0, 3.5),
        "vibration": (3.1, 0.55),
        "pressure": (4.55, 0.18),
        "speed": (1740.0, 24.0),
        "health": (62.0, 80.0),
    },
    "Critical": {
        "temperature": (101.0, 5.0),
        "vibration": (6.8, 0.80),
        "pressure": (3.65, 0.25),
        "speed": (1650.0, 38.0),
        "health": (22.0, 46.0),
    },
}


class DigitalTwin:
    """In-memory software twin that generates and stores the current machine state."""

    def __init__(self, machine_id: str) -> None:
        self.machine_id = machine_id
        self.revision = 0
        self.current_state: dict | None = None

    def update(
        self,
        condition: str,
        timestamp: datetime | None = None,
    ) -> dict:
        if condition not in _PROFILES:
            raise ValueError(f"Unsupported simulated condition: {condition}")

        profile = _PROFILES[condition]
        temperature = random.gauss(*profile["temperature"])
        vibration = random.gauss(*profile["vibration"])
        pressure = random.gauss(*profile["pressure"])
        speed = random.gauss(*profile["speed"])
        health_low, health_high = profile["health"]

        self.revision += 1
        self.current_state = {
            "time": timestamp or datetime.now().astimezone(),
            "machine_id": self.machine_id,
            "condition": condition,
            "temperature_c": round(max(20.0, temperature), 1),
            "vibration_mm_s": round(max(0.05, vibration), 2),
            "pressure_bar": round(max(0.1, pressure), 2),
            "rotational_speed_rpm": round(max(0.0, speed), 0),
            "health_score": round(random.uniform(health_low, health_high), 1),
            "revision": self.revision,
        }
        return self.current_state.copy()
