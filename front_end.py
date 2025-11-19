import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

API = "http://127.0.0.1:8000"   # change if deployed


# -------------- Utility --------------
def call_api(endpoint, params=None):
    try:
        res = requests.get(f"{API}/{endpoint}", params=params)
        return res.json()
    except Exception as e:
        return {"error": str(e)}

st.set_page_config(page_title="Medical Assessment Dashboard", layout="wide")
st.title("🧠 Medical Assessment Dashboard")


# ===============================
# Sidebar Navigation
# ===============================
page = st.sidebar.selectbox(
    "📌 Choose a Task",
    ["EDA", "Prediction", "Anomaly Detection", "Doctor Report Generator"]
)

# ================================================================
# TASK 1 → EDA PLOTS
# ================================================================
# ================================================================
# TASK 1 → EDA EXACT SAME PLOTS USING MATPLOTLIB + SEABORN
# ================================================================

if page == "EDA":
    st.header("📊 Exploratory Data Analysis (Same as Python Code)")

    if st.button("Run EDA"):
        data = call_api("eda")

        if "error" in data:
            st.error(data["error"])
        else:
            st.success(data["message"])

            # Extract the dicts
            distributions = data["distributions"]
            frequencies = data["frequencies"]

            # Tabs
            tab1, tab2 = st.tabs(["📈 Distributions", "📊 Frequencies"])

            # --------------------------------
            # TAB 1: DISTRIBUTIONS (3 PLOTS)
            # --------------------------------
            with tab1:

                # -------- AGE DISTRIBUTION --------
                st.subheader("Age Distribution")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(
                    pd.DataFrame.from_dict(distributions["age"], orient="index")
                    .reset_index()
                    .rename(columns={"index": "Age", 0: "Count"})["Age"],
                    kde=True,
                    ax=ax
                )
                st.pyplot(fig)

                # -------- BILLING AMOUNT DISTRIBUTION --------
                st.subheader("Billing Amount Distribution")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(
                    pd.DataFrame.from_dict(distributions["billing_amount"], orient="index")
                    .reset_index()
                    .rename(columns={"index": "Amount", 0: "Count"})["Amount"],
                    kde=True,
                    ax=ax
                )
                st.pyplot(fig)

                # -------- ROOM NUMBER DISTRIBUTION --------
                st.subheader("Room Number Distribution")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(
                    pd.DataFrame.from_dict(distributions["room_number"], orient="index")
                    .reset_index()
                    .rename(columns={"index": "Room", 0: "Count"})["Room"],
                    kde=True,
                    ax=ax
                )
                st.pyplot(fig)

            # --------------------------------
            # TAB 2: FREQUENCIES (3 BAR PLOTS)
            # --------------------------------
            with tab2:

                # -------- MEDICAL CONDITION --------
                st.subheader("Medical Condition Frequency")
                fig, ax = plt.subplots(figsize=(6, 4))
                pd.Series(frequencies["medical_condition"]).plot(kind="bar", ax=ax)
                st.pyplot(fig)

                # -------- ADMISSION TYPE --------
                st.subheader("Admission Type Frequency")
                fig, ax = plt.subplots(figsize=(6, 4))
                pd.Series(frequencies["admission_type"]).plot(kind="bar", ax=ax)
                st.pyplot(fig)

                # -------- MEDICATION --------
                st.subheader("Medication Frequency")
                fig, ax = plt.subplots(figsize=(6, 4))
                pd.Series(frequencies["medication"]).plot(kind="bar", ax=ax)
                st.pyplot(fig)


# ================================================================
# TASK 2 → PREDICTION MODEL RESULTS
# ================================================================
elif page == "Prediction":
    st.header("🤖 Prediction Model Results")

    if st.button("Run Prediction"):
        data = call_api("predict")

        if "error" in data:
            st.error(data["error"])
        else:
            st.subheader("📌 Model Metrics")
            st.metric("Accuracy", round(data["accuracy"], 4))
            st.metric("Precision", round(data["precision"], 4))
            st.metric("Recall", round(data["recall"], 4))
            st.metric("F1 Score", round(data["f1_score"], 4))

            st.subheader("📋 Predictions Preview (first 50)")
            df = pd.DataFrame(data["predictions_preview"])
            st.dataframe(df, use_container_width=True)



# ================================================================
# TASK 3 → anomaly detection
# ================================================================
elif page == "Anomaly Detection":
    st.header("⚠ Anomaly Detection")

    if st.button("Detect Anomalies"):
        data = call_api("anomalies")

        if "error" in data:
            st.error(data["error"])
        else:
            st.metric("Total Records", data["total_records"])
            st.metric("Anomaly Count", data["anomaly_count"])
            st.metric("Anomaly %", f"{data['anomaly_percentage']}%")

            st.subheader("🚨 Anomaly Examples (first 10)")
            df = pd.DataFrame(data["examples"])
            st.dataframe(df, use_container_width=True)



# ================================================================
# TASK 4 → Doctor Recommendation
# ================================================================
elif page == "Doctor Report Generator":
    st.header("👨‍⚕️ Doctor Recommendation")

    # fetch patient names once
    patient_data = call_api("patients")

    if "error" in patient_data:
        st.error("Failed to load patient names")
    else:
        patient_names = patient_data
        selected = st.selectbox("Select Patient Name", patient_names)

        if st.button("Generate Recommendation"):
            data = call_api("recommend", params={"patient_name": selected})

            if "error" in data:
                st.error(data["error"])
            else:
                st.subheader(f"🧑‍⚕ Recommendation for: **{data['patient_name']}**")

                st.write("### Predicted Test Result")
                st.success(data["predicted_test_result"])

                st.write("### Doctor Recommendation")
                st.info(data["recommendation"])

