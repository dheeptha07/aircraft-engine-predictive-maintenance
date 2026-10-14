import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Aircraft Engine Predictive Maintenance",
    page_icon="✈️",
    layout="wide",
)

# -----------------------------
# Dashboard styling
# -----------------------------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #F5F1E8;
            color: #292D2B;
        }
        .block-container {
            padding-top: 1.5rem;
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
        [data-testid="stExpander"] {
            background-color: #FFFCF6;
            border: 1px solid #E5DED1;
            border-radius: 10px;
        }
        [data-testid="stVegaLiteChart"] {
            background-color: #FFFCF6;
            border: 1px solid #E5DED1;
            border-radius: 12px;
            padding: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Load models and test data
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rf_model = joblib.load(os.path.join(BASE_DIR, "models", "rf_rul_model.pkl"))
isolation_forest = joblib.load(os.path.join(BASE_DIR, "models", "isolation_forest.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
useful_sensors = joblib.load(os.path.join(BASE_DIR, "models", "useful_sensors.pkl"))

test_path = os.path.join(BASE_DIR, "data", "test_FD001.txt")
columns = (
    ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)
test = pd.read_csv(test_path, sep=r"\s+", header=None)
test.columns = columns

# -----------------------------
# Fleet-level predictions
# Keep this calculation separate from the display tabs.
# -----------------------------
fleet_rows = []

for engine_id in sorted(test["engine_id"].unique()):
    fleet_engine = (
        test[test["engine_id"] == engine_id]
        .sort_values("cycle")
    )
    fleet_latest = fleet_engine.iloc[-1]

    X_fleet_latest = fleet_latest[useful_sensors].to_frame().T
    X_fleet_scaled = scaler.transform(X_fleet_latest)

    fleet_rul = max(0, float(rf_model.predict(X_fleet_scaled)[0]))

    fleet_features = fleet_engine[useful_sensors]
    fleet_features_scaled = scaler.transform(fleet_features)
    fleet_anomalies = isolation_forest.predict(fleet_features_scaled)
    fleet_anomaly_flags = (fleet_anomalies == -1).astype(int)
    recent_anomaly_rate = pd.Series(fleet_anomaly_flags).tail(10).mean()
    fleet_persistent_warning = recent_anomaly_rate >= 0.7

    if fleet_rul <= 25:
        fleet_status = "CRITICAL"
    elif fleet_persistent_warning or fleet_rul <= 50:
        fleet_status = "WARNING"
    else:
        fleet_status = "NORMAL"

    fleet_rows.append(
        {
            "Engine ID": int(engine_id),
            "Current Cycle": int(fleet_latest["cycle"]),
            "Predicted RUL": round(fleet_rul, 1),
            "Anomaly Rate (Last 10)": round(recent_anomaly_rate, 2),
            "Status": fleet_status,
        }
    )

fleet_summary = pd.DataFrame(fleet_rows)
normal_count = int((fleet_summary["Status"] == "NORMAL").sum())
warning_count = int((fleet_summary["Status"] == "WARNING").sum())
critical_count = int((fleet_summary["Status"] == "CRITICAL").sum())

# -----------------------------
# Header and navigation
# -----------------------------
st.title("✈️ Aircraft Engine Predictive Maintenance")
st.subheader("AI-powered Remaining Useful Life & Early Degradation Detection")
st.markdown(
    "This research prototype analyzes aircraft engine sensor data to estimate "
    "Remaining Useful Life (RUL) and identify persistent abnormal operating patterns."
)

engine_tab, fleet_tab, evaluation_tab = st.tabs(
    ["Engine Monitor", "Fleet Intelligence", "Model Evaluation"]
)

# ============================================================
# TAB 1: ENGINE MONITOR
# ============================================================
with engine_tab:
    st.markdown("### Individual Engine Assessment")
    st.caption(
        "Inspect an engine's latest observation, estimated remaining useful life, "
        "and recent sensor behaviour."
    )

    engine_ids = sorted(test["engine_id"].unique())
    selected_engine = st.selectbox("Select Engine", engine_ids)

    engine_data = (
        test[test["engine_id"] == selected_engine]
        .sort_values("cycle")
    )
    latest = engine_data.iloc[-1]

    # Latest-observation prediction
    X_latest = latest[useful_sensors].to_frame().T
    X_latest_scaled = scaler.transform(X_latest)
    predicted_rul = max(0, float(rf_model.predict(X_latest_scaled)[0]))

    anomaly_prediction = isolation_forest.predict(X_latest_scaled)[0]
    anomaly_score = float(isolation_forest.decision_function(X_latest_scaled)[0])

    # Engine trajectory prediction; smoothed values are for display only.
    engine_features = engine_data[useful_sensors]
    engine_features_scaled = scaler.transform(engine_features)
    engine_rul_predictions = np.maximum(
        rf_model.predict(engine_features_scaled), 0
    )
    rul_smoothed = (
        pd.Series(engine_rul_predictions)
        .rolling(window=7, min_periods=1)
        .mean()
        .values
    )

    engine_anomaly = isolation_forest.predict(engine_features_scaled)
    engine_anomaly_flag = (engine_anomaly == -1).astype(int)
    engine_anomaly_rate = (
        pd.Series(engine_anomaly_flag)
        .rolling(window=10, min_periods=10)
        .mean()
    )
    persistent_warning = bool(
        len(engine_anomaly_rate) and engine_anomaly_rate.iloc[-1] >= 0.7
    )

    if predicted_rul <= 25:
        status = "CRITICAL"
        assessment_title = "🔴 Critical condition"
        assessment_text = (
            "The predicted remaining useful life is very low. "
            "Prioritize this engine for inspection and maintenance review."
        )
    elif persistent_warning and predicted_rul > 50:
        status = "WARNING"
        assessment_title = "🟠 Early degradation detected"
        assessment_text = (
            "Persistent abnormal sensor behaviour has been detected while "
            "predicted RUL remains above the critical region. Increase monitoring "
            "and consider scheduling an inspection."
        )
    elif predicted_rul <= 50:
        status = "WARNING"
        assessment_title = "🟠 Low remaining useful life"
        assessment_text = (
            "The predicted RUL has entered the critical monitoring region. "
            "Prioritize the engine for inspection planning."
        )
    else:
        status = "NORMAL"
        assessment_title = "🟢 Normal operating condition"
        assessment_text = (
            "No persistent degradation pattern is currently detected. "
            "Continue routine monitoring."
        )

    # Compact status row
    metric_cols = st.columns(4)
    metric_cols[0].metric("CURRENT CYCLE", f"{int(latest['cycle'])}")
    metric_cols[1].metric("PREDICTED RUL", f"{predicted_rul:.0f} cycles")
    metric_cols[2].metric("ANOMALY SCORE", f"{anomaly_score:.3f}")
    metric_cols[3].metric(
    "ENGINE STATUS",
    status,
    delta={
        "NORMAL": "Routine monitoring",
        "WARNING": "Inspection recommended",
        "CRITICAL": "Priority inspection",
    }.get(status, ""),
    delta_color={
        "NORMAL": "normal",
        "WARNING": "off",
        "CRITICAL": "inverse",
    }.get(status, "off"),
)

    if status == "CRITICAL":
        st.error(f"{assessment_title} — {assessment_text}")
    elif status == "WARNING":
        st.warning(f"{assessment_title} — {assessment_text}")
    else:
        st.success(f"{assessment_title} — {assessment_text}")

    with st.container(border=True):
        st.markdown("#### Engine Health Summary")
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.write(f"**Engine ID:** {int(selected_engine)}")
            st.write(f"**Current cycle:** {int(latest['cycle'])}")
            st.write(f"**Predicted RUL:** {predicted_rul:.0f} cycles")
        with summary_col2:
            degradation_status = (
                "Persistent anomaly detected"
                if persistent_warning
                else "No persistent anomaly detected"
            )
            st.write(f"**Sensor status:** {degradation_status}")
            st.write(f"**Anomaly score:** {anomaly_score:.3f}")
            st.write(f"**Overall assessment:** {status}")

    st.divider()
    st.markdown("### RUL Degradation Trajectory")
    trajectory = pd.DataFrame(
        {
            "Cycle": engine_data["cycle"].values,
            "Predicted RUL": rul_smoothed,
            "Critical Threshold": 50,
        }
    )

    rul_line = (
        alt.Chart(trajectory)
        .mark_line(color="#B86F52", strokeWidth=2.5)
        .encode(
            x=alt.X("Cycle:Q", title="Operating Cycle"),
            y=alt.Y("Predicted RUL:Q", title="Predicted RUL (cycles)"),
            tooltip=[
                alt.Tooltip("Cycle:Q", format=".0f"),
                alt.Tooltip("Predicted RUL:Q", format=".1f"),
            ],
        )
    )
    critical_region = (
    alt.Chart(trajectory)
    .mark_area(color="#B86F52", opacity=0.12)
    .encode(
        x=alt.X("Cycle:Q"),
        y=alt.Y("Critical Threshold:Q", scale=alt.Scale(domain=[0, 125])),
    )
)
    threshold_line = (
        alt.Chart(trajectory)
        .mark_rule(color="#9A6A36", strokeDash=[6, 4], strokeWidth=1.5)
        .encode(y=alt.datum(50))
    )
    st.altair_chart(
        (critical_region + rul_line + threshold_line).properties(height=300),
        use_container_width=True,
    )
    st.caption(
    "Dashed line: 50-cycle monitoring threshold. "
    "Shaded area: critical monitoring region (predicted RUL ≤ 50 cycles)."
)

    # Sensor explanations are available when needed, rather than dominating the page.
    sensor_importance = pd.DataFrame(
        {
            "Sensor": useful_sensors,
            "Importance": rf_model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    top_sensors = sensor_importance.head(5)

    with st.expander("Sensor diagnostics and feature importance"):
        st.caption(
            "Feature importance shows which inputs the Random Forest uses most. "
            "It does not by itself establish causation."
        )
        top_sensor_table = top_sensors.copy()
        top_sensor_table["Current Value"] = [
            latest[sensor] for sensor in top_sensor_table["Sensor"]
        ]
        top_sensor_table["Importance"] = top_sensor_table["Importance"].round(3)
        top_sensor_table = top_sensor_table[
            ["Sensor", "Current Value", "Importance"]
        ]
        st.dataframe(
            top_sensor_table,
            hide_index=True,
            use_container_width=True,
        )

        importance_chart = (
            alt.Chart(top_sensors)
            .mark_bar(color="#B86F52", cornerRadiusEnd=4)
            .encode(
                x=alt.X("Importance:Q", title="Feature Importance"),
                y=alt.Y("Sensor:N", title=None, sort="-x"),
                tooltip=[
                    alt.Tooltip("Sensor:N", title="Sensor"),
                    alt.Tooltip("Importance:Q", format=".3f"),
                ],
            )
            .properties(height=200)
        )
        st.altair_chart(importance_chart, use_container_width=True)


        st.markdown("#### Recent Sensor Behaviour")
        st.caption(
            "The three most influential sensors over the latest 50 observed cycles. "
            "Each sensor is standardized independently for visualization."
        )

        sensor_columns = top_sensors.head(3)["Sensor"].tolist()
        sensor_trends = engine_data[
            ["cycle"] + sensor_columns
        ].tail(50).copy()

        # Standardize each sensor independently for visualization.
        for sensor in sensor_columns:
            values = sensor_trends[sensor].astype(float)
            sensor_std = values.std(ddof=0)

            if sensor_std > 0:
                sensor_trends[sensor] = (
                    values - values.mean()
                ) / sensor_std
            else:
                sensor_trends[sensor] = 0.0

        
        chart_cols = st.columns(len(sensor_columns))

        for col, sensor in zip(chart_cols, sensor_columns):
            with col:
                st.markdown(
                    f"<div style='font-size:16px; font-weight:600; "
                    f"padding: 8px 0 10px 0; "
                    f"color: #243447;'>{sensor}</div>",
                    unsafe_allow_html=True
                )

                sensor_chart = (
                    alt.Chart(sensor_trends)
                    .mark_line(
                        color="#B86F52",
                        strokeWidth=2.5,
                        point=True
                    )
                    .encode(
                        x=alt.X(
                            "cycle:Q",
                            title="Operating Cycle",
                            axis=alt.Axis(tickMinStep=5)
                        ),
                        y=alt.Y(
                            f"{sensor}:Q",
                            title="Standardized Reading",
                            scale=alt.Scale(zero=False)
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "cycle:Q",
                                title="Cycle",
                                format=".0f"
                            ),
                            alt.Tooltip(
                                f"{sensor}:Q",
                                title="Standardized Reading",
                                format=".2f"
                            ),
                        ],
                    )
                    .properties(height=220)
                )

                st.altair_chart(
                    sensor_chart,
                    use_container_width=True
                )


        st.caption(
            "Sensor values are standardized independently for visualization. "
            "These charts help interpret sensor behaviour but do not change "
            "the model prediction."
        )


# ============================================================
# TAB 2: FLEET INTELLIGENCE
# ============================================================
with fleet_tab:
    st.markdown("### Fleet Overview")
    st.caption(
        "Predicted conditions across the 100 NASA C-MAPSS FD001 test engines. "
        "Statuses are prototype triage indicators, not certified maintenance decisions."
    )

    fleet_metrics = st.columns(4)
    fleet_metrics[0].metric("TOTAL ENGINES", len(fleet_summary))
    fleet_metrics[1].metric("NORMAL", normal_count)
    fleet_metrics[2].metric("WARNING", warning_count)
    fleet_metrics[3].metric("CRITICAL", critical_count)

    attention_engines = (
        fleet_summary[
            fleet_summary["Status"].isin(["WARNING", "CRITICAL"])
        ]
        .sort_values("Predicted RUL")
    )

    st.markdown("#### Engines Requiring Attention")
    if attention_engines.empty:
        st.success("No engines currently meet the warning thresholds.")
    else:
        st.dataframe(
            attention_engines,
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("View all engine predictions"):
        st.dataframe(
            fleet_summary.sort_values("Predicted RUL"),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Fleet status distribution"):
        status_distribution = pd.DataFrame(
            {
                "Status": ["NORMAL", "WARNING", "CRITICAL"],
                "Engine Count": [normal_count, warning_count, critical_count],
            }
        )
        fleet_chart = (
            alt.Chart(status_distribution)
            .mark_bar(color="#B86F52", cornerRadiusEnd=4)
            .encode(
                x=alt.X(
                    "Status:N",
                    title="Engine Status",
                    sort=["NORMAL", "WARNING", "CRITICAL"],
                ),
                y=alt.Y("Engine Count:Q", title="Number of Engines"),
                tooltip=[
                    alt.Tooltip("Status:N"),
                    alt.Tooltip("Engine Count:Q", format=".0f"),
                ],
            )
            .properties(height=220)
        )
        st.altair_chart(fleet_chart, use_container_width=True)

    st.caption(
        "A warning is based on predicted RUL thresholds and the proportion of "
        "anomalies detected in the latest 10 observations."
    )

# ============================================================
# TAB 3: MODEL EVALUATION
# ============================================================
with evaluation_tab:
    st.markdown("### Model Performance & Comparison")
    st.caption(
        "Validation experiments and the final capped-RUL model's unseen-test results. "
        "The capped-RUL test result uses a different target formulation and should "
        "not be treated as a direct like-for-like comparison with the validation rows."
    )

    model_results = pd.DataFrame(
        {
            "Model": [
                "KNN",
                "KNN + PCA",
                "Random Forest",
                "Random Forest + PCA",
                "Random Forest (RUL capped)",
            ],
            "MAE": [28.83, 28.92, 26.28, 26.77, 11.92],
            "RMSE": [39.51, 39.74, 35.90, 36.92, 17.40],
            "R²": [0.638, 0.634, 0.701, 0.684, 0.812],
            "Evaluation": [
                "Validation",
                "Validation",
                "Validation",
                "Validation",
                "Unseen test set",
            ],
        }
    )

    st.dataframe(model_results, hide_index=True, use_container_width=True)

    st.markdown("#### MAE Comparison")
    mae_chart = (
        alt.Chart(model_results)
        .mark_bar(color="#B86F52", cornerRadiusEnd=4)
        .encode(
            x=alt.X("MAE:Q", title="Mean Absolute Error (cycles)"),
            y=alt.Y("Model:N", title=None, sort="-x"),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip("MAE:Q", format=".2f"),
                alt.Tooltip("Evaluation:N"),
            ],
        )
        .properties(height=220)
    )
    st.altair_chart(mae_chart, use_container_width=True)

    st.markdown("#### Final Model — Unseen Test Performance")
    final_metrics = st.columns(3)
    final_metrics[0].metric("MAE", "11.92 cycles")
    final_metrics[1].metric("RMSE", "17.40 cycles")
    final_metrics[2].metric("R²", "0.812")
    st.caption(
        "Final model: Random Forest trained with RUL capped at 125 cycles. "
        "The final test metrics use the capped target formulation."
    )

    st.divider()
    st.markdown("### How the System Works")
    st.markdown(
        """
        **1. Sensor data**  
        Engine operating measurements are collected across multiple cycles.

        **2. Data preprocessing**  
        Low-variance sensors are removed and retained sensor values are standardized.

        **3. RUL prediction**  
        A Random Forest model estimates Remaining Useful Life (RUL).

        **4. Anomaly detection**  
        An Isolation Forest identifies observations that differ from learned operating patterns.

        **5. Persistent warning detection**  
        A warning is raised when abnormal behaviour persists across recent cycles.

        **6. Maintenance assessment**  
        RUL estimates and anomaly signals are combined into a simple monitoring status.
        """
    )

    st.info(
        "Research prototype using simulated NASA C-MAPSS data. It is intended for "
        "experimentation and demonstration, not as a certified aviation maintenance system."
    )
