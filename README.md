# Digital Twin-Based Predictive Maintenance for Industrial IoT Using Deep Learning

A Streamlit college-project demonstration for a simulated industrial turbine.
It runs without physical sensors, hardware, a database, or external services.

## What it demonstrates

- TURBINE_01 digital-twin state updates with changing temperature, vibration,
  pressure, rotational-speed, and health readings.
- Healthy, Degrading, and Critical simulation controls.
- Historical sensor charts and machine status history.
- A configurable maintenance-alert threshold and rule-based remaining useful
  life (RUL) estimate.
- Maintenance recommendations based on the selected condition and sensor values.
- A network-monitor view with adjustable simulated message delay and recorded
  processing, transport, and total message times.

## Important model note

Despite “Using Deep Learning” in the project title, this demonstration does **not**
include a trained deep-learning model. The RUL value is a transparent,
rule-based estimate calculated from simulated condition and sensor indicators.
It is not validated against failure data and must not be used to make decisions
about real equipment. A trained model would require an appropriate historical
degradation dataset, a defined prediction target, training, and separate
evaluation.

## Run locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run artifacts/turbine-maintenance/app.py
```

Then open the local URL printed by Streamlit (usually
`http://localhost:8501`). Use the sidebar to change the machine condition, RUL
alert threshold, and simulated network delay.

## Run the checks

```bash
cd artifacts/turbine-maintenance
python -m unittest discover -s tests
python -m compileall -q app.py src tests
```

## Run in this Replit project

The project preview starts the same Streamlit app through its configured web
workflow. The Python dependencies are also listed in `requirements.txt`.
