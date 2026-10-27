from __future__ import annotations

import math

_CONDITION_RISK_FLOOR = {
    "Healthy": 0.05,
    "Degrading": 0.45,
    "Critical": 0.82,
}


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(value, maximum))


def estimate_rul_hours(reading: dict) -> float:
    """Return a transparent, simulated rule-based RUL estimate in hours.

    This is deliberately not a trained model and is only suitable for a demo.
    """
    temperature_stress = _clamp((reading["temperature_c"] - 65.0) / 45.0)
    vibration_stress = _clamp(reading["vibration_mm_s"] / 8.0)
    pressure_stress = _clamp(abs(reading["pressure_bar"] - 5.2) / 2.2)
    health_stress = _clamp(1.0 - reading["health_score"] / 100.0)

    sensor_risk = (
        0.30 * temperature_stress
        + 0.30 * vibration_stress
        + 0.15 * pressure_stress
        + 0.25 * health_stress
    )
    risk_score = max(
        _CONDITION_RISK_FLOOR[reading["condition"]],
        _clamp(sensor_risk),
    )
    return round(max(1.0, 360.0 * math.exp(-4.0 * risk_score)), 1)


def get_maintenance_recommendations(reading: dict) -> list[str]:
    recommendations: list[str] = []
    condition = reading["condition"]

    if condition == "Critical":
        recommendations.append(
            "Demonstration response: stop the simulated turbine and inspect it "
            "before returning it to service."
        )
    elif condition == "Degrading":
        recommendations.append(
            "Schedule a preventive inspection during the next planned maintenance window."
        )

    if reading["vibration_mm_s"] >= 5.0:
        recommendations.append(
            "Inspect bearings, shaft alignment, and lubrication for vibration-related wear."
        )
    elif reading["vibration_mm_s"] >= 2.0:
        recommendations.append(
            "Trend vibration and check bearing lubrication at the next inspection."
        )

    if reading["temperature_c"] >= 95.0:
        recommendations.append(
            "Check cooling flow and inspect for friction or overheating."
        )

    if reading["pressure_bar"] < 4.0:
        recommendations.append(
            "Inspect the simulated pressure circuit, seals, and pressure sensor."
        )

    if condition == "Healthy" and not recommendations:
        recommendations.append(
            "Continue routine monitoring and follow the normal inspection schedule."
        )

    return recommendations
