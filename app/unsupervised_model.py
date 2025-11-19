import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import joblib
import os
from dotenv import load_dotenv
load_dotenv()
DATA_PATH = os.getenv('data_path')
MODEL_PATH = "models/task3_anomaly_model.pkl"

# =========================================================
# 🔥 TRAINING – Identify Normal vs Abnormal Billing Patterns
# =========================================================
def train_task3_model():
    print("\n===== TRAINING TASK-3 ANOMALY MODEL =====\n")

    df = pd.read_csv(DATA_PATH)

    if "Billing Amount" not in df.columns:
        return {"error": "'Billing Amount' column missing in dataset"}

    # Use only numeric billing values
    billing = df["Billing Amount"].values.reshape(-1, 1)

    # Train Isolation Forest
    model = IsolationForest(
        contamination=0.05,     # Detect top 5% as anomalies
        n_estimators=200,
        random_state=42
    )
    model.fit(billing)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Model Saved:", MODEL_PATH)
    print("\n===== TRAINING COMPLETE =====\n")

    return {"message": "Task-3 anomaly detection model trained successfully!"}


# =========================================================
# 🔥 DETECTION – Identify abnormal billing records
# =========================================================
def detect_task3_anomalies():
    print("\n===== RUNNING ANOMALY DETECTION =====\n")

    df = pd.read_csv(DATA_PATH)

    if "Billing Amount" not in df.columns:
        return {"error": "'Billing Amount' column missing in dataset"}

    model = joblib.load(MODEL_PATH)

    # Predict anomalies
    df["Anomaly"] = model.predict(df[["Billing Amount"]])  # -1 = anomaly, 1 = normal

    # Generate interpretation using Z-score
    mean_val = df["Billing Amount"].mean()
    std_val = df["Billing Amount"].std()

    def interpret(amount):
        z = (amount - mean_val) / std_val
        if z > 2:
            return "Exceptionally high billing amount"
        elif z < -2:
            return "Unusually low billing amount"
        else:
            return "Mild deviation from normal pattern"

    # Add interpretation column
    df["Interpretation"] = df.apply(
        lambda row: interpret(row["Billing Amount"]) if row["Anomaly"] == -1 else "Normal billing",
        axis=1
    )

    anomalies = df[df["Anomaly"] == -1]

    print("Anomalies Detected:", len(anomalies))

    return {
        "total_records": len(df),
        "anomaly_count": len(anomalies),
        "anomaly_percentage": round((len(anomalies) / len(df)) * 100, 2),
        "examples": anomalies.head(10).to_dict(orient="records")
    }


# ===========================
# 🔥 RUN (For Testing Only)
# ===========================
# train_task3_model()
# print(detect_task3_anomalies())
