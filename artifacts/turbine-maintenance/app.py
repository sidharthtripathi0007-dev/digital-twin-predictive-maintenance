import json
from pathlib import Path
from src.twin import CONDITIONS, DigitalTwin
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from src.maintenance import estimate_rul_hours, get_maintenance_recommendations
from src.lstm_inference import load_heldout_predictions, parse_cmapss_csv
from src.lstm_model import NumpyLSTMRegressor
from src.network import simulate_message_latency

MACHINE_ID = "TURBINE_01"
# ---------------- LSTM MODEL CONFIGURATION ----------------

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model_artifacts"

LSTM_MODEL_PATH = MODEL_DIR / "fd001_lstm.npz"
PREPROCESSING_PATH = MODEL_DIR / "preprocessing.json"

@st.cache_resource
def load_lstm_model():
    """Load the saved NumPy LSTM and its preprocessing configuration."""

    if not LSTM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"LSTM checkpoint not found: {LSTM_MODEL_PATH}"
        )

    if not PREPROCESSING_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessing configuration not found: {PREPROCESSING_PATH}"
        )

    with PREPROCESSING_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    input_size = int(config["input_size"])
    hidden_size = int(config["hidden_size"])
    window_size = int(config["window_size"])

    if input_size != 24 or window_size != 30:
        raise ValueError(
            f"Unexpected model configuration: "
            f"{input_size} features, {window_size} cycles."
        )

    if len(config["feature_columns"]) != input_size:
        raise ValueError("Feature column count does not match input_size.")

    if len(config["means"]) != input_size:
        raise ValueError("Normalization means have the wrong length.")

    if len(config["scales"]) != input_size:
        raise ValueError("Normalization scales have the wrong length.")

    model = NumpyLSTMRegressor.load(
        LSTM_MODEL_PATH,
        input_size=input_size,
        hidden_size=hidden_size,
    )

    return model, config
def predict_lstm_rul(sequence, model, config):
    """
    Predict RUL from 30 consecutive C-MAPSS cycles.

    sequence must contain 30 rows and 24 features in the
    exact order specified by preprocessing.json.
    """

    import numpy as np

    expected_shape = (
        int(config["window_size"]),
        int(config["input_size"]),
    )

    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.shape != expected_shape:
        raise ValueError(
            f"Expected {expected_shape} input, "
            f"but received {sequence.shape}."
        )

    if not np.isfinite(sequence).all():
        raise ValueError("Input sequence contains invalid values.")

    means = np.asarray(config["means"], dtype=np.float32)
    scales = np.asarray(config["scales"], dtype=np.float32)

    if np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("Invalid normalization scales.")

    normalized = (sequence - means) / scales

    prediction = float(
        model.predict(normalized[np.newaxis, :, :])[0]
    )

    if not np.isfinite(prediction):
        raise ValueError("The LSTM returned a non-finite prediction.")

    target_scale = float(config["target_scale"])
    rul_cap = float(config["rul_cap_cycles"])

    # Training uses RUL divided by the target scale.
    rul_cycles = float(np.clip(prediction * target_scale, 0, rul_cap))

    return rul_cycles


@st.cache_data(show_spinner=False)
def _load_lstm_demo_data():
    report = None
    try:
        if (MODEL_DIR / "training_report.json").exists():
            with (MODEL_DIR / "training_report.json").open(
                "r",
                encoding="utf-8",
            ) as report_file:
                report = json.load(report_file)
        if not report or report.get("status") != "trained":
            return {
                "report": report,
                "predictions": None,
                "error": "No completed NASA FD001 LSTM training report is available.",
            }

        model, config = load_lstm_model()
        report, predictions = load_heldout_predictions(
            MODEL_DIR,
            model,
            config,
        )
        return {
            "report": report,
            "predictions": predictions,
            "error": None,
        }
    except Exception as error:
        return {
            "report": report,
            "predictions": None,
            "error": str(error),
        }


HISTORY_LIMIT = 180

st.set_page_config(
    page_title=(
        "Digital Twin-Based Predictive Maintenance for Industrial IoT Using Deep Learning"
    ),
    layout="wide",
    initial_sidebar_state="expanded",
)


def _initialize_demo_state() -> None:
    if "lstm_predicted_rul" not in st.session_state:
        st.session_state["lstm_predicted_rul"] = None
    if "lstm_prediction_source" not in st.session_state:
        st.session_state["lstm_prediction_source"] = None
    if "_last_lstm_demo_engine" not in st.session_state:
        st.session_state["_last_lstm_demo_engine"] = None

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
            "TURBINE_01 readings and message timings are generated locally. "
            "The NASA C-MAPSS LSTM is a separate benchmark and does not use "
            "these simulated readings."
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

    st.subheader("Separate NASA FD001 LSTM result")
    latest_lstm_rul = st.session_state.get("lstm_predicted_rul")
    if latest_lstm_rul is None:
        st.info(
            "No successful LSTM inference is available yet. Open the LSTM Model "
            "tab to view the FD001 benchmark or upload a compatible C-MAPSS CSV."
        )
    else:
        st.metric(
            "Latest NASA C-MAPSS LSTM RUL",
            f"{latest_lstm_rul:.1f} cycles",
        )
        st.caption(
            f"Source: {st.session_state.get('lstm_prediction_source') or 'NASA FD001'}. "
            "This cycle estimate is separate from TURBINE_01's simulated "
            "hour-based heuristic."
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
    "Demonstration only: TURBINE_01 telemetry and its hour-based RUL alerts are "
    "simulated. A separately trained NASA FD001 LSTM benchmark is available in "
    "the LSTM Model tab; it does not consume TURBINE_01 readings."
)
overview_tab, sensors_tab, history_tab, network_tab, lstm_tab = st.tabs(
    [
        "Overview",
        "Sensor history",
        "Machine status history",
        "Network monitor",
        "LSTM Model",
    ]
)

@st.fragment(run_every="3s")
def _render_live_dashboard() -> None:
    reading = _record_live_sample()
    
    
    with overview_tab:
        _render_overview(reading)
    with sensors_tab:
        _render_sensor_history()
    with history_tab:
        _render_machine_history()
    with network_tab:
        _render_network_monitor()


def _render_lstm_model_tab() -> None:
    st.subheader("LSTM Model Prediction")

    demo = _load_lstm_demo_data()
    report = demo.get("report")
    predictions = demo.get("predictions")

    if report and report.get("status") == "trained":
        st.success(
            "Trained NASA C-MAPSS FD001 LSTM loaded from the saved model artifacts."
        )
        rmse = report.get("held_out_rmse_cycles")
        mae = report.get("held_out_mae_cycles")
        if rmse is not None and mae is not None:
            metric_columns = st.columns(3)
            metric_columns[0].metric("Held-out RMSE", f"{float(rmse):.4f} cycles")
            metric_columns[1].metric("Held-out MAE", f"{float(mae):.3f} cycles")
            metric_columns[2].metric(
                "Held-out engines",
                str(report.get("held_out_engine_count", "—")),
            )
        st.caption(
            f"Engine-disjoint evaluation over "
            f"{report.get('held_out_window_count', '—')} held-out windows; "
            f"RUL targets capped at {report.get('training_config', {}).get('rul_cap_cycles', '—')} cycles."
        )
        st.info(
            "The saved model input is 24 values per cycle in preprocessing.json: "
            "3 C-MAPSS operating settings plus 21 sensor channels, over 30 "
            "consecutive cycles. It is not compatible with TURBINE_01's four "
            "simulated readings."
        )
    elif demo.get("error"):
        st.error(f"LSTM model status: {demo['error']}")

    if predictions is not None and not predictions.empty:
        st.markdown("**Predictions on held-out NASA FD001 engines**")
        engine_ids = predictions["engine_id"].astype(int).tolist()
        selected_engine = st.selectbox(
            "Inspect a held-out engine",
            options=engine_ids,
            format_func=lambda engine_id: f"Engine {engine_id}",
            key="lstm_demo_engine_id",
        )
        selected = predictions.loc[
            predictions["engine_id"] == selected_engine
        ].iloc[0]
        prediction_columns = st.columns(3)
        prediction_columns[0].metric(
            "LSTM predicted RUL",
            f"{selected['predicted_rul_cycles']:.1f} cycles",
        )
        prediction_columns[1].metric(
            "Known held-out RUL",
            f"{selected['actual_rul_cycles']:.0f} cycles",
        )
        prediction_columns[2].metric(
            "Absolute error",
            f"{selected['absolute_error_cycles']:.1f} cycles",
        )
        if st.session_state.get("_last_lstm_demo_engine") != selected_engine:
            st.session_state["lstm_predicted_rul"] = float(
                selected["predicted_rul_cycles"]
            )
            st.session_state["lstm_prediction_source"] = (
                f"NASA FD001 held-out engine {selected_engine}"
            )
            st.session_state["_last_lstm_demo_engine"] = selected_engine

        display = predictions[
            [
                "engine_id",
                "cycle",
                "predicted_rul_cycles",
                "actual_rul_cycles",
                "absolute_error_cycles",
            ]
        ].rename(
            columns={
                "engine_id": "Engine",
                "cycle": "Cycle",
                "predicted_rul_cycles": "LSTM predicted RUL (cycles)",
                "actual_rul_cycles": "Known RUL (cycles)",
                "absolute_error_cycles": "Absolute error (cycles)",
            }
        )
        for column in (
            "LSTM predicted RUL (cycles)",
            "Absolute error (cycles)",
        ):
            display[column] = display[column].map(lambda value: f"{value:.2f}")
        st.dataframe(display, width="stretch", hide_index=True)
    elif report and report.get("status") == "trained":
        st.warning(
            "The training report is available, but the saved held-out sequence "
            f"demo could not be loaded: {demo.get('error') or 'no prediction rows'}"
        )

    st.markdown("**Run inference on a C-MAPSS CSV**")
    st.caption(
        "Upload rows with the exact model feature names, or raw C-MAPSS rows. "
        "The predictor selects the latest 30 cycles per engine, sorts and checks "
        "cycle continuity when cycle numbers are included, and otherwise uses "
        "the row order in the file."
    )
    uploaded_csv = st.file_uploader(
        "Upload C-MAPSS CSV",
        type=["csv", "txt"],
        key="lstm_csv",
    )

    if uploaded_csv is not None:
        try:
            model, config = load_lstm_model()
            sequences = parse_cmapss_csv(uploaded_csv.getvalue(), config)
            if len(sequences) > 1:
                labels = [sequence["label"] for sequence in sequences]
                selected_label = st.selectbox(
                    "Choose an uploaded engine sequence",
                    options=labels,
                    key="lstm_uploaded_engine",
                )
                selected_sequence = next(
                    item["sequence"]
                    for item in sequences
                    if item["label"] == selected_label
                )
            else:
                selected_label = sequences[0]["label"]
                selected_sequence = sequences[0]["sequence"]

            rul = predict_lstm_rul(selected_sequence, model, config)
            st.success("LSTM inference completed on the uploaded C-MAPSS sequence.")
            st.metric("Uploaded sequence predicted RUL", f"{rul:.1f} cycles")
            st.session_state["lstm_predicted_rul"] = rul
            st.session_state["lstm_prediction_source"] = (
                f"Uploaded {uploaded_csv.name} · {selected_label}"
            )

        except Exception as exc:
            st.error(f"LSTM inference error: {exc}")


with lstm_tab:
    _render_lstm_model_tab()

_render_live_dashboard()