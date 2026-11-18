import os
import joblib
import numpy as np
import pandas as pd

def test_artifacts_exist():
    assert os.path.exists("failure_model.joblib")
    assert os.path.exists("sensor_telemetry.csv")
    assert os.path.exists("telemetry_analysis.png")

def test_model_inference_probability():
    model = joblib.load("failure_model.joblib")
    sample = pd.DataFrame([{
        "air_temp_k": 302.0,
        "process_temp_k": 313.5,
        "rotational_speed_rpm": 1400,
        "torque_nm": 65.0,
        "tool_wear_min": 210.0,
        "temp_diff_k": 11.5,
        "power_watts": 65.0 * (1400 * 2 * np.pi / 60),
    }])
    prob = model.predict_proba(sample)[0][1]
    assert 0.0 <= prob <= 1.0
    assert prob > 0.5
