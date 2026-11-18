import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

def build_and_train():
    np.random.seed(42)
    n_samples = 2000
    air_temp = np.random.normal(300, 2.0, n_samples)
    process_temp = air_temp + np.random.normal(10, 1.0, n_samples)
    rpm = np.random.normal(1500, 180, n_samples)
    torque = np.random.normal(40, 9.5, n_samples)
    tool_wear = np.random.uniform(0, 240, n_samples)

    risk_score = (
        0.03 * (torque - 40)
        + 0.015 * (tool_wear - 120)
        + 0.4 * ((process_temp - air_temp) - 10)
        + np.random.normal(0, 0.5, n_samples)
    )
    failure = (risk_score > 1.2).astype(int)

    df = pd.DataFrame({
        "air_temp_k": np.round(air_temp, 2),
        "process_temp_k": np.round(process_temp, 2),
        "rotational_speed_rpm": np.round(rpm, 1),
        "torque_nm": np.round(torque, 2),
        "tool_wear_min": np.round(tool_wear, 1),
        "machine_failure": failure,
    })

    df["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]
    df["power_watts"] = df["torque_nm"] * (df["rotational_speed_rpm"] * 2 * np.pi / 60)
    df.to_csv("sensor_telemetry.csv", index=False)

    healthy_torque = df[df["machine_failure"] == 0]["torque_nm"]
    failed_torque = df[df["machine_failure"] == 1]["torque_nm"]
    t_stat, p_val = stats.ttest_ind(failed_torque, healthy_torque, equal_var=False)
    print(f"Welch's T-Test (Torque): t={t_stat:.4f}, p={p_val:.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(df["tool_wear_min"], df["torque_nm"], c=df["machine_failure"], cmap="coolwarm", alpha=0.6, edgecolors="k", linewidth=0.3)
    axes[0].set_title("Torque vs. Tool Wear (Red = Failure)")
    axes[0].set_xlabel("Tool Wear (min)")
    axes[0].set_ylabel("Torque (Nm)")

    axes[1].hist(df[df["machine_failure"] == 0]["power_watts"], bins=30, alpha=0.6, label="Healthy")
    axes[1].hist(df[df["machine_failure"] == 1]["power_watts"], bins=30, alpha=0.6, label="Failure")
    axes[1].set_title("Mechanical Power Distribution")
    axes[1].set_xlabel("Power (Watts)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("telemetry_analysis.png", dpi=150)
    plt.close()

    features = ["air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min", "temp_diff_k", "power_watts"]
    X, y = df[features], df["machine_failure"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("ROC-AUC Score:", round(roc_auc_score(y_test, y_prob), 4))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "failure_model.joblib")
    print("Artifacts saved successfully!")

if __name__ == "__main__":
    build_and_train()
