import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import altair as alt


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Aircraft Engine Predictive Maintenance",
    page_icon="✈️",
    layout="wide"
)
# -----------------------------
# Custom dashboard styling
# -----------------------------


st.markdown("""
<style>
    .stApp {
        background-color: #F5F1E8;
        color: #292D2B;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1350px;
    }

    h1 {
        color: #292D2B;
        font-size: 2rem;
        font-weight: 750;
        letter-spacing: -0.7px;
    }

    h2, h3 {
        color: #393C38;
        font-weight: 650;
        letter-spacing: -0.3px;
    }

    p, label, li {
        color: #4B4F49;
    }

    [data-testid="stMetric"] {
        background-color: #FFFCF6;
        border: 1px solid #E5DED1;
        padding: 16px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #77786F;
        font-size: 0.8rem;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #292D2B;
        font-weight: 700;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #E5DED1;
        border-radius: 12px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #E5DED1;
        border-radius: 10px;
        overflow: hidden;
    }

    div[data-baseweb="select"] > div {
        background-color: #FFFCF6;
        border-color: #D6CDBE;
        border-radius: 8px;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: #E5DED1;
    }

    [data-testid="stCaptionContainer"] {
        color: #77786F;
    }

    [data-testid="stExpander"] {
        background-color: #FFFCF6;
        border: 1px solid #E5DED1;
        border-radius: 10px;
    }

    /* Chart containers */
    [data-testid="stArrowVegaLiteChart"],
    [data-testid="stVegaLiteChart"] {
        background-color: #FFFCF6;
        border: 1px solid #E5DED1;
        border-radius: 12px;
        padding: 8px;
    }
</style>
""", unsafe_allow_html=True)




# -----------------------------
# Load models
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rf_model = joblib.load(
    os.path.join(BASE_DIR, "models", "rf_rul_model.pkl")
)

isolation_forest = joblib.load(
    os.path.join(BASE_DIR, "models", "isolation_forest.pkl")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "models", "scaler.pkl")
)

useful_sensors = joblib.load(
    os.path.join(BASE_DIR, "models", "useful_sensors.pkl")
)


# -----------------------------
# Load test data
# -----------------------------

test_path = os.path.join(
    BASE_DIR,
    "data",
    "test_FD001.txt"
)

columns = (
    ["engine_id", "cycle",
     "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

test = pd.read_csv(
    test_path,
    sep=r"\s+",
    header=None
)

test.columns = columns

# -----------------------------
# Fleet-level predictions
# -----------------------------

fleet_rows = []

for engine_id in sorted(test["engine_id"].unique()):
    fleet_engine = (
        test[test["engine_id"] == engine_id]
        .sort_values("cycle")
    )

    # Use the latest observed cycle for this engine
    fleet_latest = fleet_engine.iloc[-1]

    X_fleet_latest = fleet_latest[useful_sensors].to_frame().T
    X_fleet_scaled = scaler.transform(X_fleet_latest)

    fleet_rul = max(
        0,
        float(rf_model.predict(X_fleet_scaled)[0])
    )

    # Check for persistent anomalies across the latest 10 cycles
    fleet_features = fleet_engine[useful_sensors]
    fleet_features_scaled = scaler.transform(fleet_features)

    fleet_anomalies = isolation_forest.predict(
        fleet_features_scaled
    )
    fleet_anomaly_flags = (fleet_anomalies == -1).astype(int)

    recent_anomaly_rate = pd.Series(
        fleet_anomaly_flags
    ).tail(10).mean()

    fleet_persistent_warning = recent_anomaly_rate >= 0.7

    if fleet_rul <= 25:
        fleet_status = "CRITICAL"
    elif fleet_persistent_warning or fleet_rul <= 50:
        fleet_status = "WARNING"
    else:
        fleet_status = "NORMAL"

    fleet_rows.append({
        "Engine ID": int(engine_id),
        "Current Cycle": int(fleet_latest["cycle"]),
        "Predicted RUL": round(fleet_rul, 1),
        "Anomaly Rate (Last 10)": round(recent_anomaly_rate, 2),
        "Status": fleet_status
    })

fleet_summary = pd.DataFrame(fleet_rows)



# -----------------------------
# Header
# -----------------------------

st.title("✈️ Aircraft Engine Predictive Maintenance")
st.subheader("AI-powered Remaining Useful Life & Early Degradation Detection")

st.markdown(
    """
    This research prototype analyzes aircraft engine sensor data
    to estimate Remaining Useful Life (RUL) and identify persistent
    abnormal operating patterns.
    """
)
# Main dashboard navigation
engine_tab, fleet_tab, evaluation_tab = st.tabs([
    "Engine Monitor",
    "Fleet Intelligence",
    "Model Evaluation"
])


# -----------------------------
# Engine selection
# -----------------------------

engine_ids = sorted(test["engine_id"].unique())

selected_engine = st.selectbox(
    "Select Engine",
    engine_ids
)


# -----------------------------
# Get latest engine observation
# -----------------------------

engine_data = test[
    test["engine_id"] == selected_engine
].sort_values("cycle")

latest = engine_data.iloc[-1]


# -----------------------------
# RUL Prediction
# -----------------------------

X_latest = latest[useful_sensors].to_frame().T

X_latest_scaled = scaler.transform(X_latest)

predicted_rul = rf_model.predict(
    X_latest_scaled
)[0]

predicted_rul = max(0, predicted_rul)

# -----------------------------
# Sensor importance
# -----------------------------

sensor_importance = pd.DataFrame({
    "Sensor": useful_sensors,
    "Importance": rf_model.feature_importances_
})

sensor_importance = sensor_importance.sort_values(
    "Importance",
    ascending=False
)

top_sensors = sensor_importance.head(5)
# -----------------------------
# Anomaly Detection
# -----------------------------

anomaly_prediction = isolation_forest.predict(
    X_latest_scaled
)[0]

anomaly_score = isolation_forest.decision_function(
    X_latest_scaled
)[0]

# -----------------------------
# RUL trajectory
# -----------------------------

engine_features = engine_data[useful_sensors]

engine_features_scaled = scaler.transform(
    engine_features
)

engine_rul_predictions = rf_model.predict(
    engine_features_scaled
)

engine_rul_predictions = np.maximum(
    engine_rul_predictions,
    0
)
# Smooth predictions for visualization only
rul_series = pd.Series(engine_rul_predictions)

rul_smoothed = (
    rul_series
    .rolling(window=7, min_periods=1)
    .mean()
    .values
)

# -----------------------------
# Sensor-level explanation
# -----------------------------

# -----------------------------
# Sensor-level explanation
# -----------------------------

st.divider()

st.subheader("🔍 Key Sensor Indicators")

st.caption(
    "Top sensor variables influencing the Random Forest RUL prediction."
)

top_sensor_table = top_sensors.copy()

top_sensor_table["Current Value"] = [
    latest[sensor] for sensor in top_sensor_table["Sensor"]
]

top_sensor_table["Importance"] = (
    top_sensor_table["Importance"].round(3)
)

top_sensor_table = top_sensor_table[
    ["Sensor", "Current Value", "Importance"]
]

st.dataframe(
    top_sensor_table,
    hide_index=True,
    use_container_width=True
)


importance_chart = alt.Chart(top_sensors).mark_bar(
    color="#B86F52",
    cornerRadiusEnd=4
).encode(
    x=alt.X(
        "Importance:Q",
        title="Feature Importance"
    ),
    y=alt.Y(
        "Sensor:N",
        title=None,
        sort="-x"
    ),
    tooltip=[
        alt.Tooltip("Sensor:N", title="Sensor"),
        alt.Tooltip("Importance:Q", format=".3f")
    ]
).properties(
    height=220
)

st.altair_chart(
    importance_chart,
    use_container_width=True
)


# -----------------------------
# Persistent degradation warning
# -----------------------------

# -----------------------------
# Recent sensor behavior
# -----------------------------

st.divider()

st.subheader("📈 Recent Sensor Behavior")

st.caption(
    "Recent behavior of the three most influential sensors "
    "for the selected engine."
)

sensor_columns = top_sensors.head(3)["Sensor"].tolist()

sensor_trends = engine_data[
    ["cycle"] + sensor_columns
].tail(50).copy()

# Normalize each sensor for visualization
for sensor in sensor_columns:
    sensor_mean = sensor_trends[sensor].mean()
    sensor_std = sensor_trends[sensor].std()

    if sensor_std != 0:
        sensor_trends[sensor] = (
            sensor_trends[sensor] - sensor_mean
        ) / sensor_std

# Display each sensor separately
chart_cols = st.columns(3)

for col, sensor in zip(chart_cols, sensor_columns):
    with col:
        st.markdown(f"**{sensor}**")
        
        
sensor_chart = alt.Chart(
    sensor_trends
).mark_line(
    color="#B86F52",
    strokeWidth=2
).encode(
    x=alt.X("cycle:Q", title="Operating Cycle"),
    y=alt.Y(sensor, type="quantitative", title="Standardized Reading"),
    tooltip=[
        alt.Tooltip("cycle:Q", title="Cycle", format=".0f"),
        alt.Tooltip(sensor, title="Standardized Reading", format=".2f")
    ]
).properties(
    height=180
)

st.altair_chart(sensor_chart, use_container_width=True)


st.caption(
    "Values are standardized independently for visualization. "
    "Positive values indicate readings above the sensor's average, "
    "while negative values indicate readings below its average. "
    "These trends are used for interpretation and do not change the model prediction."
)

engine_anomaly = isolation_forest.predict(
    engine_features_scaled
)

engine_anomaly_flag = (
    engine_anomaly == -1
).astype(int)

engine_anomaly_rate = (
    pd.Series(engine_anomaly_flag)
    .rolling(window=10, min_periods=10)
    .mean()
)

persistent_warning = (
    engine_anomaly_rate.iloc[-1] >= 0.7
)

# -----------------------------
# Status
# -----------------------------

if predicted_rul <= 25:
    status = "🔴 CRITICAL"
elif persistent_warning:
    status = "🟠 WARNING"
elif predicted_rul <= 50:
    status = "🟠 WARNING"
else:
    status = "🟢 NORMAL"

# -----------------------------
# Maintenance assessment
# -----------------------------

st.subheader("🔧 Maintenance Assessment")

if predicted_rul <= 25:
    assessment_title = "🔴 Critical condition"
    assessment_text = (
        "The predicted remaining useful life is very low. "
        "Prioritize this engine for inspection and maintenance review."
    )

elif persistent_warning and predicted_rul > 50:
    assessment_title = "🟠 Early degradation detected"
    assessment_text = (
        "Persistent abnormal sensor behavior has been detected "
        "while predicted RUL remains above the critical region. "
        "Increase monitoring and consider scheduling an inspection."
    )

elif predicted_rul <= 50:
    assessment_title = "🟠 Low remaining useful life"
    assessment_text = (
        "The predicted RUL has entered the critical monitoring region. "
        "Prioritize the engine for inspection planning."
    )

else:
    assessment_title = "🟢 Normal operating condition"
    assessment_text = (
        "No persistent degradation pattern is currently detected. "
        "Continue routine monitoring."
    )

st.markdown(f"### {assessment_title}")
st.write(assessment_text)


# -----------------------------
# Engine Health Summary
# -----------------------------

st.divider()

st.subheader("✈️ Engine Health Summary")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.markdown("#### Engine Information")
    st.write(f"**Engine ID:** {int(selected_engine)}")
    st.write(f"**Current Cycle:** {int(latest['cycle'])}")
    st.write(f"**Predicted RUL:** {predicted_rul:.0f} cycles")

with summary_col2:
    st.markdown("#### Degradation Assessment")

    if persistent_warning:
        degradation_status = "🟠 Persistent anomaly detected"
    else:
        degradation_status = "🟢 No persistent anomaly detected"

    st.write(f"**Sensor Status:** {degradation_status}")
    st.write(f"**Anomaly Score:** {anomaly_score:.3f}")

    if predicted_rul <= 25:
        overall_assessment = "🔴 Critical monitoring region"
    elif persistent_warning or predicted_rul <= 50:
        overall_assessment = "🟠 Increased monitoring recommended"
    else:
        overall_assessment = "🟢 No current warning triggered"

    st.write(f"**Overall Assessment:** {overall_assessment}")

# -----------------------------
# Fleet Overview
# -----------------------------

st.divider()

st.subheader("✈️ Fleet Overview")

st.caption(
    "Summary of predicted engine conditions across the "
    "100 NASA C-MAPSS FD001 test engines."
)

normal_count = (fleet_summary["Status"] == "NORMAL").sum()
warning_count = (fleet_summary["Status"] == "WARNING").sum()
critical_count = (fleet_summary["Status"] == "CRITICAL").sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Engines", len(fleet_summary))

with col2:
    st.metric("🟢 Normal", int(normal_count))

with col3:
    st.metric("🟠 Warning", int(warning_count))

with col4:
    st.metric("🔴 Critical", int(critical_count))

st.markdown("### Engines Requiring Attention")

attention_engines = (
    fleet_summary[
        fleet_summary["Status"].isin(["WARNING", "CRITICAL"])
    ]
    .sort_values("Predicted RUL")
)

if attention_engines.empty:
    st.success("No engines currently meet the warning thresholds.")
else:
    st.dataframe(
        attention_engines,
        hide_index=True,
        use_container_width=True
    )

with st.expander("View all engine predictions"):
    st.dataframe(
        fleet_summary.sort_values("Predicted RUL"),
        hide_index=True,
        use_container_width=True
    )

# Fleet status distribution

st.markdown("### Fleet Status Distribution")

status_distribution = pd.DataFrame({
    "Status": ["NORMAL", "WARNING", "CRITICAL"],
    "Engine Count": [
        int(normal_count),
        int(warning_count),
        int(critical_count)
    ]
})


fleet_chart = alt.Chart(status_distribution).mark_bar(
    color="#B86F52",
    cornerRadiusEnd=4
).encode(
    x=alt.X("Status:N", title="Engine Status", sort=["NORMAL", "WARNING", "CRITICAL"]),
    y=alt.Y("Engine Count:Q", title="Number of Engines"),
    tooltip=[
        alt.Tooltip("Status:N"),
        alt.Tooltip("Engine Count:Q", format=".0f")
    ]
).properties(
    height=280
)

st.altair_chart(fleet_chart, use_container_width=True)


st.caption(
    "Counts are based on the prototype's predicted RUL "
    "and persistent-anomaly thresholds."
)


# -----------------------------
# Dashboard metrics
# -----------------------------

# -----------------------------
# Engine status panel
# -----------------------------

st.subheader("Engine Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container(border=True):
        st.caption("CURRENT CYCLE")
        st.markdown(f"### {int(latest['cycle'])}")
        st.caption("Operating cycles")

with col2:
    with st.container(border=True):
        st.caption("PREDICTED RUL")
        st.markdown(f"### {predicted_rul:.0f}")
        st.caption("Remaining cycles")

with col3:
    with st.container(border=True):
        st.caption("ANOMALY SCORE")
        st.markdown(f"### {anomaly_score:.3f}")
        st.caption("Isolation Forest")

with col4:
    with st.container(border=True):
        st.caption("ENGINE STATUS")

        if predicted_rul <= 25:
            st.markdown("### 🔴 CRITICAL")
        elif persistent_warning:
            st.markdown("### 🟠 WARNING")
        elif predicted_rul <= 50:
            st.markdown("### 🟠 WARNING")
        else:
            st.markdown("### 🟢 NORMAL")

        st.caption("Current assessment")
    st.divider()

if predicted_rul <= 25:
    st.error(
        "🔴 CRITICAL — Predicted RUL is very low. "
        "Immediate inspection is recommended."
    )
elif persistent_warning:
    st.warning(
        "🟠 EARLY DEGRADATION WARNING — "
        "Persistent abnormal sensor behavior has been detected "
        "across recent operating cycles."
    )
elif predicted_rul <= 50:
    st.warning(
        "🟠 RUL WARNING — "
        "Predicted remaining useful life has entered the "
        "critical monitoring region."
    )
else:
    st.success(
        "🟢 NORMAL — No persistent degradation pattern detected."
    )


# -----------------------------
# Engine information
# -----------------------------

st.divider()

st.subheader(
    f"Engine {selected_engine} — Current Assessment"
)

if persistent_warning:
    st.warning(
        "Persistent abnormal sensor behavior detected "
        "across the recent operating cycles."
    )
elif anomaly_prediction == -1:
    st.info(
        "An abnormal observation was detected, "
        "but persistent degradation has not been confirmed."
    )
else:
    st.success(
        "No persistent degradation pattern detected "
        "in the recent operating cycles."
    )

# -----------------------------
# RUL trajectory visualization
# -----------------------------

# -----------------------------
# RUL trajectory visualization
# -----------------------------

st.divider()

st.subheader("📈 RUL Degradation Trajectory")

trajectory = pd.DataFrame({
    "Cycle": engine_data["cycle"].values,
    "Predicted RUL": rul_smoothed,
    "Critical Threshold": 50
})


rul_chart = alt.Chart(trajectory).mark_line(
    color="#B86F52",
    strokeWidth=2.5
).encode(
    x=alt.X("Cycle:Q", title="Operating Cycle"),
    y=alt.Y("Predicted RUL:Q", title="Predicted RUL (cycles)"),
    tooltip=[
        alt.Tooltip("Cycle:Q", format=".0f"),
        alt.Tooltip("Predicted RUL:Q", format=".1f")
    ]
).properties(
    height=300
)

st.altair_chart(rul_chart, use_container_width=True)


st.caption(
    "The 50-cycle threshold marks the critical degradation region."
)

# -----------------------------
# Model performance
# -----------------------------

st.divider()

st.subheader("📊 Model Performance")


# -----------------------------
# Model Performance & Comparison
# -----------------------------

st.divider()

st.subheader("📊 Model Performance & Comparison")

st.caption(
    "Comparison of experiments conducted during model development."
)

model_results = pd.DataFrame({
    "Model": [
        "KNN",
        "KNN + PCA",
        "Random Forest",
        "Random Forest + PCA",
        "Random Forest (RUL capped)"
    ],
    "MAE": [
        28.83,
        28.92,
        26.28,
        26.77,
        11.92
    ],
    "RMSE": [
        39.51,
        39.74,
        35.90,
        36.92,
        17.40
    ],
    "R²": [
        0.638,
        0.634,
        0.701,
        0.684,
        0.812
    ],
    "Evaluation": [
        "Validation",
        "Validation",
        "Validation",
        "Validation",
        "Unseen test set"
    ]
})

st.dataframe(
    model_results,
    hide_index=True,
    use_container_width=True
)

st.markdown("### MAE Comparison")

st.caption(
    "Lower MAE indicates a smaller average prediction error. "
    "The capped-RUL model was evaluated under a different target "
    "formulation on the unseen test set."
)


mae_chart = alt.Chart(model_results).mark_bar(
    color="#B86F52",
    cornerRadiusEnd=4
).encode(
    x=alt.X("MAE:Q", title="Mean Absolute Error (cycles)"),
    y=alt.Y("Model:N", title=None, sort="-x"),
    tooltip=[
        alt.Tooltip("Model:N"),
        alt.Tooltip("MAE:Q", format=".2f")
    ]
).properties(
    height=240
)

st.altair_chart(mae_chart, use_container_width=True)


st.markdown("### Final Model — Unseen Test Performance")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("MAE", "11.92 cycles")

with metric2:
    st.metric("RMSE", "17.40 cycles")

with metric3:
    st.metric("R²", "0.812")

st.caption(
    "Final model: Random Forest trained with RUL capped at 125 cycles. "
    "The final test metrics use the capped target formulation."
)


# -----------------------------
# How the system works
# -----------------------------

st.divider()

st.subheader("⚙️ How the System Works")

st.markdown("""
**1. Sensor Data**  
Aircraft engine operating data is collected across multiple cycles.

**2. Data Preprocessing**  
Low-variance sensors are removed and the remaining sensor values are standardized.

**3. RUL Prediction**  
A Random Forest model estimates the engine's Remaining Useful Life (RUL).

**4. Anomaly Detection**  
An Isolation Forest identifies abnormal sensor behavior compared with healthy operating patterns.

**5. Persistent Warning Detection**  
A warning is triggered when abnormal behavior persists across recent operating cycles.

**6. Maintenance Assessment**  
RUL predictions and degradation signals are combined to provide a simple maintenance-oriented status.
""")