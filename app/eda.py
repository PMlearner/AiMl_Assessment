# services/task1_eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

load_dotenv()
DATA_PATH = os.getenv("data_path")

def run_eda():
    df = pd.read_csv(DATA_PATH)

    # plt.figure(figsize=(6, 4))
    # sns.histplot(df["Age"], kde=True)
    # plt.title("Age Distribution")

    # plt.figure(figsize=(6, 4))
    # sns.histplot(df["Billing Amount"], kde=True)
    # plt.title("Billing Amount Distribution")

    # plt.figure(figsize=(6, 4))
    # sns.histplot(df["Room Number"], kde=True)
    # plt.title("Room Number Distribution")

    # plt.figure(figsize=(6, 4))
    # df["Medical Condition"].value_counts().plot(kind="bar")
    # plt.title("Medical Condition Frequency")

    # plt.figure(figsize=(6, 4))
    # df["Admission Type"].value_counts().plot(kind="bar")
    # plt.title("Admission Type Frequency")

    # plt.figure(figsize=(6, 4))
    # df["Medication"].value_counts().plot(kind="bar")
    # plt.title("Medication Frequency")

    # ⭐ Show all windows at the same time
    plt.show()

    plt.close()

    # ---------------------- RETURN DATA ----------------------

    age_dist = df["Age"].describe().to_dict()
    billing_dist = df["Billing Amount"].describe().to_dict()
    room_dist = df["Room Number"].describe().to_dict()

    condition_freq = df["Medical Condition"].value_counts().to_dict()
    admission_freq = df["Admission Type"].value_counts().to_dict()
    medication_freq = df["Medication"].value_counts().to_dict()

    return {
        "status": "success",
        "message": "EDA completed (plots generated but not saved).",
        "distributions": {
            "age": age_dist,
            "billing_amount": billing_dist,
            "room_number": room_dist
        },
        "frequencies": {
            "medical_condition": condition_freq,
            "admission_type": admission_freq,
            "medication": medication_freq
        }
    }


# Debug
# print(run_eda())
