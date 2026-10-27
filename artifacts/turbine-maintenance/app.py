from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from src.maintenance import estimate_rul_hours, get_maintenance_recommendations
from src.network import simulate_message_latency
from src.twin import CONDITIONS, DigitalTwin

MACHINE_ID = "TURBINE_01"
HISTORY_LIMIT = 180

st.set_page_config(
    page_title=(
        "Digital Twin-Based Predictive Maintenance for Industrial IoT Using Deep Learning"
    ),
    layout="wide",
    initial_sidebar_state="expanded",
)


def _initialize_demo_state() -> None:
    if "twin" in st.session_state:
        return

    twin = DigitalTwin(MACHINE_ID)
    now = datetime.now().astimezone()
    sensor_history: list[dict] = []
    status_history: list[dict] = []
    network_history: list[dict] = []

    # Seed an explicitly synthetic hour of telemetry so charts are useful on
    # the first page load, before the live update loop has accumulated samples.
    for minutes_ago in range(30, -1, -1):
        sample_time = now - timedelta(minutes=minutes_ago * 2)
        reading = twin.update("Healthy", timestamp=sample_time)
        reading["rul_hours"] = estimate_rul_hours(reading)
        sensor_history.append(reading)
        status_history.append(
            {
                "time": sample_time,
                "status": reading["condition"],
                "health": reading["health_score"],
                "rul_hours": reading["rul_hours"],
                "event": "Synthetic telemetry sample",
            }
        )
        network_history.append(
            {
                "time": sample_time,
                **simulate_message_latency(configured_delay_ms=40),
            }
        )

    st.session_state.twin = twin
    st.session_state.active_condition = "Healthy"
    st.session_state.sensor_history = sensor_history[-HISTORY_LIMIT:]
    st.session_state.status_history = status_history[-HISTORY_LIMIT:]
    st.session_state.network_history = network_history[-HISTORY_LIMIT:]
    st.session_state.rul_alert_threshold = 72
    st.session_state.simulated_network_delay_ms = 40
    st.session_state.last_reading = sensor_history[-1]


def _append_limited(key: str, entry: dict) -> None:
    entries = st.session_state[key]
    entries.append(entry)
    st.session_state[key] = entries[-HISTORY_LIMIT:]


def _set_condition(condition: str) -> None:
    st.session_state.active_condition = condition


def _record_live_sample() -> dict:
    now = datetime.now().astimezone()
    reading = st.session_state.twin.update(
        st.session_state.active_condition,
        timestamp=now,
    )
    reading["rul_hours"] = estimate_rul_hours(reading)

    previous = st.session_state.last_reading
    event = "Telemetry update"
    if reading["condition"] != previous["condition"]:
        event = f"Condition changed: {previous['condition']} → {reading['condition']}"

    _append_limited("sensor_history", reading)
    _append_limited(
        "status_history",
        {
            "time": now,
            "status": reading["condition"],
            "health": reading["health_score"],
            "rul_hours": reading["rul_hours"],
            "event": event,
        },
    )
    _append_limited(
        "network_history",
        {
            "time": now,
            **simulate_message_latency(
                configured_delay_ms=st.session_state.simulated_network_delay_ms
            ),
        },
    )
    st.session_state.last_reading = reading
    return reading


def _format_history(entries: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(entries)
    if frame.empty:
        return frame
    frame["time"] = pd.to_datetime(frame["time"]).dt.strftime("%H:%M:%S")
    return frame


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Simulation controls")
        st.caption(f"Machine ID · `{MACHINE_ID}`")

        st.markdown("**Select a machine condition**")
        for condition in CONDITIONS:
            is_active = st.session_state.active_condition == condition
            if st.button(
                f"{'● ' if is_active else ''}{condition}",
                key=f"condition-{condition.lower()}",
                width="stretch",
                type="primary" if is_active else "secondary",
                help=f"Switch simulated sensor behavior to {condition.lower()} conditions.",
            ):
                _set_condition(condition)
                st.rerun()

        st.divider()
        st.session_state.rul_alert_threshold = st.slider(
            "Maintenance alert threshold (hours)",
            min_value=12,
            max_value=240,
            value=int(st.session_state.rul_alert_threshold),
            step=12,
            help="Show an alert whenever the simulated RUL estimate is below this value.",
        )
        st.session_state.simulated_network_delay_ms = st.slider(
            "Simulated network delay (ms)",
            min_value=0,
            max_value=2000,
            value=int(st.session_state.simulated_network_delay_ms),
            step=50,
            help="Adds a configurable simulated transport delay to each telemetry message.",
        )

        st.divider()
        st.caption("Simulation only")
        st.caption(
            "No physical sensors, machine connection, or trained model are used. "
            "All readings and message timings are generated locally."
        )


def _render_health_status(reading: dict) -> None:
    condition = reading["condition"]
    if condition == "Critical":
        st.error("CRITICAL CONDITION · Immediate inspection recommended")
    elif condition == "Degrading":
        st.warning("DEGRADING CONDITION · Plan preventive maintenance")
    else:
        st.success("HEALTHY CONDITION · Operating within the demo's normal range")


def _render_recommendations(reading: dict) -> None:
    recommendations = get_maintenance_recommendations(reading)
    if not recommendations:
        st.info("No maintenance actions are currently suggested.")
        return
    for recommendation in recommendations:
        st.markdown(f"- {recommendation}")


def _render_overview(reading: dict) -> None:
    _render_health_status(reading)

    if reading["rul_hours"] < st.session_state.rul_alert_threshold:
        st.error(
            f"Automatic maintenance alert: estimated remaining useful life is "
            f"{reading['rul_hours']:.1f} hours, below your "
            f"{st.session_state.rul_alert_threshold}-hour threshold."
        )
    else:
        st.info(
            f"No RUL alert: estimate is above the "
            f"{st.session_state.rul_alert_threshold}-hour threshold."
        )

    st.subheader("Live machine state")
    st.caption(
        f"Digital twin updated {reading['time'].strftime('%H:%M:%S %Z')} · "
        f"State revision {reading['revision']:,}"
    )
    metric_columns = st.columns(4)
    metrics = (
        ("Temperature", f"{reading['temperature_c']:.1f} °C"),
        ("Vibration", f"{reading['vibration_mm_s']:.2f} mm/s"),
        ("Pressure", f"{reading['pressure_bar']:.2f} bar"),
        ("Rotational speed", f"{reading['rotational_speed_rpm']:.0f} rpm"),
    )
    for column, (label, value) in zip(metric_columns, metrics):
        column.metric(label, value)

    health_col, rul_col = st.columns(2)
    with health_col:
        st.metric("Machine health score", f"{reading['health_score']:.0f} / 100")
        st.progress(
            int(reading["health_score"]),
            text="Simulated health indicator",
        )
    with rul_col:
        st.metric(
            "Estimated remaining useful life",
            f"{reading['rul_hours']:.1f} hours",
            help="Rule-based demo estimate from the simulated condition and readings.",
        )
        st.caption("Simulated heuristic estimate · not produced by a trained model")

    with st.expander("How the RUL demo estimate works"):
        st.write(
            "The demo combines the selected condition with normalized temperature, "
            "vibration, pressure, and health indicators to produce a risk score. "
            "That score is mapped to an estimated number of hours using a simple "
            "exponential rule. This is not a deep-learning prediction, is not "
            "trained on historical failure data, and must not be used for real "
            "equipment decisions."
        )

    st.subheader("Maintenance recommendations")
    _render_recommendations(reading)


def _render_sensor_history() -> None:
    history = pd.DataFrame(st.session_state.sensor_history)
    if history.empty:
        st.info("Sensor history will appear after the first reading.")
        return

    st.caption(
        "Synthetic readings from this browser session. New readings are added "
        "automatically every few seconds."
    )
    columns = st.columns(2)
    sensor_charts = (
        ("Temperature (°C)", "temperature_c"),
        ("Vibration (mm/s)", "vibration_mm_s"),
        ("Pressure (bar)", "pressure_bar"),
        ("Rotational speed (rpm)", "rotational_speed_rpm"),
    )
    for index, (label, field) in enumerate(sensor_charts):
        with columns[index % 2]:
            st.markdown(f"**{label}**")
            chart_data = history.set_index("time")[[field]]
            st.line_chart(chart_data, height=220, y_label=label)


def _render_machine_history() -> None:
    history = _format_history(st.session_state.status_history)
    if history.empty:
        st.info("Machine status history will appear after the first update.")
        return

    st.caption("Most recent simulated twin updates and condition changes.")
    visible_columns = ["time", "status", "health", "rul_hours", "event"]
    view = history[visible_columns].tail(25).iloc[::-1].copy()
    view = view.rename(
        columns={
            "time": "Time",
            "status": "Condition",
            "health": "Health score",
            "rul_hours": "Estimated RUL (hours)",
            "event": "Event",
        }
    )
    view["Estimated RUL (hours)"] = view["Estimated RUL (hours)"].map(
        lambda value: f"{value:.1f}"
    )
    view["Health score"] = view["Health score"].map(lambda value: f"{value:.0f}")
    st.dataframe(view, width="stretch", hide_index=True)


def _render_network_monitor() -> None:
    history = pd.DataFrame(st.session_state.network_history)
    if history.empty:
        st.info("Network metrics will appear after the first simulated message.")
        return

    latest = history.iloc[-1]
    average_latency = history["total_ms"].tail(30).mean()
    max_latency = history["total_ms"].tail(30).max()
    columns = st.columns(3)
    columns[0].metric("Latest message time", f"{latest['total_ms']:.1f} ms")
    columns[1].metric("30-message average", f"{average_latency:.1f} ms")
    columns[2].metric("30-message maximum", f"{max_latency:.1f} ms")

    st.caption(
        "Network timings are simulated locally. The configured delay is applied "
        "to a generated message-time estimate; no external network is contacted."
    )
    chart_data = history.set_index("time")[["processing_ms", "transport_ms", "total_ms"]]
    st.line_chart(
        chart_data,
        height=280,
        y_label="Simulated time (ms)",
    )

    recent = _format_history(st.session_state.network_history).tail(20).iloc[::-1]
    table = recent[["time", "processing_ms", "transport_ms", "total_ms"]].rename(
        columns={
            "time": "Time",
            "processing_ms": "Processing (ms)",
            "transport_ms": "Simulated delay (ms)",
            "total_ms": "Total (ms)",
        }
    )
    for column in ("Processing (ms)", "Simulated delay (ms)", "Total (ms)"):
        table[column] = table[column].map(lambda value: f"{value:.1f}")
    st.dataframe(table, width="stretch", hide_index=True)


_initialize_demo_state()
_render_sidebar()

st.title(
    "Digital Twin-Based Predictive Maintenance for Industrial IoT Using Deep Learning"
)
st.caption(
    f"College project demonstration · software digital twin · machine `{MACHINE_ID}`"
)
st.warning(
    "Demonstration only: all telemetry is simulated. This version uses a "
    "transparent rule-based RUL estimate; no deep-learning model has been trained."
)


@st.fragment(run_every="3s")
def _render_live_dashboard() -> None:
    reading = _record_live_sample()
    overview_tab, sensors_tab, history_tab, network_tab = st.tabs(
        ["Overview", "Sensor history", "Machine status history", "Network monitor"]
    )
    with overview_tab:
        _render_overview(reading)
    with sensors_tab:
        _render_sensor_history()
    with history_tab:
        _render_machine_history()
    with network_tab:
        _render_network_monitor()


_render_live_dashboard()
