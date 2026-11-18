import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Predictive Maintenance ML Engine", layout="centered")
st.title("Predictive Equipment Failure & Telemetry Engine")
st.caption("Real-time ML inference powered by NumPy, Pandas, SciPy & Scikit-learn")

model = joblib.load("failure_model.joblib")

col1, col2 = st.columns(2)
with col1:
    air_temp = st.slider("Air Temperature (K)", 295.0, 306.0, 300.0)
    process_temp = st.slider("Process Temperature (K)", 305.0, 316.0, 310.0)
    rpm = st.slider("Rotational Speed (RPM)", 1100, 2200, 1500)
with col2:
    torque = st.slider("Torque (Nm)", 15.0, 75.0, 42.0)
    tool_wear = st.slider("Tool Wear (min)", 0, 240, 110)

temp_diff = process_temp - air_temp
power_watts = torque * (rpm * 2 * np.pi / 60)

input_df = pd.DataFrame([{
    "air_temp_k": air_temp,
    "process_temp_k": process_temp,
    "rotational_speed_rpm": rpm,
    "torque_nm": torque,
    "tool_wear_min": tool_wear,
    "temp_diff_k": temp_diff,
    "power_watts": power_watts,
}])

prob = model.predict_proba(input_df)[0][1]
st.metric("Predicted Failure Probability", f"{prob * 100:.2f}%")

if prob > 0.5:
    st.error("HIGH RISK: Immediate preventive maintenance recommended.")
else:
    st.success("NORMAL OPERATION: Telemetry within safe operating thresholds.")

with st.expander("View Telemetry Statistical Analysis"):
    st.image("telemetry_analysis.png", width="stretch")
