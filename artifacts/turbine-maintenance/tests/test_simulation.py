import unittest

from src.maintenance import estimate_rul_hours, get_maintenance_recommendations
from src.network import simulate_message_latency
from src.twin import CONDITIONS, DigitalTwin


class DigitalTwinTests(unittest.TestCase):
    def test_updates_current_state_and_changes_sensor_readings(self) -> None:
        twin = DigitalTwin("TURBINE_01")
        readings = [twin.update("Healthy") for _ in range(12)]

        self.assertEqual(twin.current_state["machine_id"], "TURBINE_01")
        self.assertEqual(readings[-1]["revision"], 12)
        self.assertGreater(len({reading["temperature_c"] for reading in readings}), 1)
        self.assertGreater(len({reading["vibration_mm_s"] for reading in readings}), 1)
        self.assertGreater(len({reading["pressure_bar"] for reading in readings}), 1)
        self.assertGreater(
            len({reading["rotational_speed_rpm"] for reading in readings}),
            1,
        )

    def test_each_requested_condition_is_supported(self) -> None:
        twin = DigitalTwin("TURBINE_01")
        for condition in CONDITIONS:
            reading = twin.update(condition)
            self.assertEqual(reading["condition"], condition)
            self.assertGreaterEqual(reading["health_score"], 0)
            self.assertLessEqual(reading["health_score"], 100)

    def test_simulated_rul_drops_as_condition_worsens(self) -> None:
        readings = {
            "Healthy": {
                "condition": "Healthy",
                "temperature_c": 69.0,
                "vibration_mm_s": 0.6,
                "pressure_bar": 5.3,
                "health_score": 95.0,
            },
            "Degrading": {
                "condition": "Degrading",
                "temperature_c": 83.0,
                "vibration_mm_s": 3.1,
                "pressure_bar": 4.5,
                "health_score": 70.0,
            },
            "Critical": {
                "condition": "Critical",
                "temperature_c": 101.0,
                "vibration_mm_s": 6.8,
                "pressure_bar": 3.6,
                "health_score": 35.0,
            },
        }
        rul = {
            condition: estimate_rul_hours(reading)
            for condition, reading in readings.items()
        }

        self.assertGreater(rul["Healthy"], rul["Degrading"])
        self.assertGreater(rul["Degrading"], rul["Critical"])
        self.assertGreater(rul["Critical"], 0)

    def test_critical_reading_includes_urgent_recommendations(self) -> None:
        critical_reading = {
            "condition": "Critical",
            "temperature_c": 101.0,
            "vibration_mm_s": 6.8,
            "pressure_bar": 3.6,
            "health_score": 35.0,
        }
        recommendations = get_maintenance_recommendations(critical_reading)

        self.assertGreaterEqual(len(recommendations), 3)
        self.assertTrue(
            any("stop the simulated turbine" in item for item in recommendations)
        )

    def test_network_delay_is_recorded_in_total_message_time(self) -> None:
        no_delay = simulate_message_latency(0)
        configured_delay = simulate_message_latency(1000)

        self.assertGreater(configured_delay["transport_ms"], 1000)
        self.assertGreater(configured_delay["total_ms"], no_delay["total_ms"])
        self.assertAlmostEqual(
            configured_delay["total_ms"],
            configured_delay["processing_ms"] + configured_delay["transport_ms"],
            places=1,
        )


if __name__ == "__main__":
    unittest.main()
