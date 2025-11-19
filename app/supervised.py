# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# import joblib
# import os
# path=r"C:\Users\VH0000051\Desktop\aiml_proj\healthcare_dataset.csv"
# # df = pd.read_csv("dataset.csv")

# # ============================
# #  API 1 — TRAIN MODEL
# # ============================
# def train_task2_model():
#     print("\n========== TASK-2 MODEL TRAINING STARTED ==========\n")

#     print("[1] Loading dataset...")
#     df = pd.read_csv(path)
#     print("    ✓ Dataset loaded successfully.")
#     print("    Shape:", df.shape)
#     print(df.head(), "\n")

#     encoders = {}

#     # 1️⃣ Convert date columns
#     print("[2] Converting date columns to numeric format...")
#     date_cols = ["Date of Admission", "Discharge Date"]
#     for col in date_cols:
#         print(f"    - Processing date column: {col}")
#         df[col] = pd.to_datetime(df[col], errors="coerce")
#         df[col] = df[col].astype("int64") // 10**9   # UNIX timestamp
#     print("    ✓ Date conversion completed.\n")

#     # 2️⃣ Encode ALL CATEGORICAL COLUMNS except target
#     print("[3] Encoding categorical features...")
#     categorical_cols = df.select_dtypes(include="object").columns.tolist()
#     categorical_cols.remove("Test Results")

#     for col in categorical_cols:
#         print(f"    - Encoding column: {col}")
#         enc = LabelEncoder()
#         df[col] = enc.fit_transform(df[col])
#         encoders[col] = enc
#     print("    ✓ All categorical columns encoded.\n")

#     # 3️⃣ Encode target column
#     print("[4] Encoding target column: Test Results")
#     target_enc = LabelEncoder()
#     df["Test Results"] = target_enc.fit_transform(df["Test Results"])
#     encoders["Test Results"] = target_enc
#     print("    ✓ Target column encoded.\n")

#     # 4️⃣ Train/test split
#     print("[5] Splitting dataset into train/test...")
#     X = df.drop(["Test Results"], axis=1)
#     y = df["Test Results"]

#     X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

#     print("    Train size:", len(X_train))
#     print("    Test size:", len(y) - len(X_train))
#     print("    ✓ Data split completed.\n")

#     # 5️⃣ Train model
#     print("[6] Training RandomForest model...")
#     model = RandomForestClassifier()

#     model.fit(X_train, y_train)
#     print("    ✓ Model training completed.\n")

#     # 6️⃣ Save model
#     print("[7] Saving model and encoders...")
#     os.makedirs("models", exist_ok=True)
#     joblib.dump(model, "models/task2_model.pkl")
#     joblib.dump(encoders, "models/task2_encoders.pkl")
#     print("    ✓ Model saved as models/task2_model.pkl")
#     print("    ✓ Encoders saved as models/task2_encoders.pkl\n")

#     print("========== TRAINING COMPLETE ==========\n")

#     return {
#         "message": "Task-2 model training completed successfully!",
#         "train_size": len(X_train)
#     }
# # ============================
# #  API 2 — TEST MODEL
# # ============================
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# def test_task2_model():
#     print("\n========== TASK-2 MODEL TESTING ==========\n")

#     df = pd.read_csv(path)

#     print("[1] Loading model and encoders...")
#     model = joblib.load("models/task2_model.pkl")
#     encoders = joblib.load("models/task2_encoders.pkl")

#     # ---------------- FIX HERE ----------------
#     print("[2] Converting date columns to numeric format (same as training)...")
#     date_cols = ["Date of Admission", "Discharge Date"]
#     for col in date_cols:
#         print(f"    - Converting: {col}")
#         df[col] = pd.to_datetime(df[col], errors="coerce")
#         df[col] = df[col].astype("int64") // 10**9
#     print("    ✓ Date conversion completed.\n")
#     # ------------------------------------------

#     print("[3] Applying encoders...")
#     for col, enc in encoders.items():
#         df[col] = enc.transform(df[col])

#     y = df["Test Results"].astype(int)

#     X = df.drop(["Test Results"], axis=1)

#     print("[4] Splitting test data...")
#     _, X_test, _, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     print("[5] Making predictions...")
#     preds = model.predict(X_test)

#     print("[6] Evaluating model...")

#     accuracy = accuracy_score(y_test, preds)
#     precision = precision_score(y_test, preds, average="weighted", zero_division=0)
#     recall = recall_score(y_test, preds, average="weighted", zero_division=0)
#     f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
#     matrix = confusion_matrix(y_test, preds)
#     report = classification_report(y_test, preds, zero_division=0)

#     print("\n===== MODEL METRICS =====")
#     print("Accuracy:", accuracy)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     print("F1 Score:", f1)
#     print("\nConfusion Matrix:\n", matrix)
#     print("\nClassification Report:\n", report)

#     print("========== TESTING COMPLETE ==========\n")

#     return {
#         "total_test_samples": len(X_test),
#         "accuracy": accuracy,
#         "precision": precision,
#         "recall": recall,
#         "f1_score": f1,
#         "confusion_matrix": matrix.tolist(),
#         "predictions": preds[:20].tolist(),
#         "actual": y_test[:20].tolist()
#     }



# # ================
# # 🚀 RUN TEST
# # ================
# # train_task2_model()
# print(test_task2_model())



#===================================================================================================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os

DATA_PATH = r"C:\Users\VH0000051\Desktop\aiml_proj\healthcare_dataset.csv"
MODEL_PATH = "models/task2_model.cbm"
TARGET_ENCODER_PATH = "models/task2_target_encoder.pkl"
CAT_COLS_PATH = "models/task2_cat_columns.pkl"
# ============================================================
# 🔥 1. TRAIN MODEL (Optimized CatBoost)
# ============================================================
def train_task2_model():

    print("\n========== TRAINING TASK-2 MODEL ==========\n")

    df = pd.read_csv(DATA_PATH)
    print("[1] Dataset Loaded — Shape:", df.shape)

    # -------------------------------------------------
    # REMOVE COLUMNS THAT SHOULD NOT BE USED FOR PREDICTION
    # -------------------------------------------------
    useless_cols = [
        "Name", "Doctor", "Hospital", "Room Number",
        "Insurance Provider", "Admission Type", "Patient ID"
    ]
    df = df.drop(columns=[c for c in useless_cols if c in df.columns])
    print("[2] Removed non-predictive columns.\n")

    # -------------------------------------------------
    # DATE CONVERSION
    # -------------------------------------------------
    print("[3] Converting date columns...")
    date_cols = ["Date of Admission", "Discharge Date"]

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = df[col].astype("int64") // 10**9
    print("    ✓ Date conversion completed.\n")

    # -------------------------------------------------
    # TARGET ENCODING
    # -------------------------------------------------
    print("[4] Encoding target column: Test Results")
    target = "Test Results"
    target_encoder = LabelEncoder()
    df[target] = target_encoder.fit_transform(df[target])
    print("    Classes:", list(target_encoder.classes_), "\n")

    # -------------------------------------------------
    # CATEGORICAL COLUMNS (Auto-detect)
    # -------------------------------------------------
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != target]

    print("[5] Categorical Columns:", categorical_cols, "\n")

    # -------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # -------------------------------------------------
    print("[6] Splitting data...")
    X = df.drop([target], axis=1)
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("    Train Size:", len(X_train))
    print("    Validation Size:", len(X_val), "\n")

    # -------------------------------------------------
    # CATBOOST MODEL
    # -------------------------------------------------
    print("[7] Training CatBoost model...")

    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=10,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_strength=2,
        l2_leaf_reg=3,
        bootstrap_type="Bayesian",
        bagging_temperature=1.2,
        od_wait=100,
        verbose=100
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        cat_features=categorical_cols
    )

    print("    ✓ Model trained.\n")

    # -------------------------------------------------
    # SAVE MODEL + ENCODERS
    # -------------------------------------------------
    os.makedirs("models", exist_ok=True)

    model.save_model(MODEL_PATH)
    joblib.dump(target_encoder, TARGET_ENCODER_PATH)
    joblib.dump(categorical_cols, CAT_COLS_PATH)

    print("[8] Model + encoders saved.\n")
    print("========== TRAINING COMPLETE ==========\n")

    return {"message": "Training completed successfully!"}

# ============================================================
# 🔥 2. TEST MODEL (Optimized CatBoost)
# ============================================================
def test_task2_model():

    print("\n========== TESTING TASK-2 MODEL ==========\n")

    df = pd.read_csv(DATA_PATH)

    # Remove useless columns
    useless_cols = [
        "Name", "Doctor", "Hospital", "Room Number",
        "Insurance Provider", "Admission Type", "Patient ID"
    ]
    df = df.drop(columns=[c for c in useless_cols if c in df.columns])

    # Load model + encoders
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    target_encoder = joblib.load(TARGET_ENCODER_PATH)
    categorical_cols = joblib.load(CAT_COLS_PATH)

    # Convert date columns
    date_cols = ["Date of Admission", "Discharge Date"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[col] = df[col].astype("int64") // 10**9

    # Encode target
    df["Test Results"] = target_encoder.transform(df["Test Results"])

    # Split dataset
    X = df.drop(["Test Results"], axis=1)
    y = df["Test Results"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # ---------------------------
    # 🔮 PREDICT
    # ---------------------------
    preds = model.predict(X_test)

    # Decode categorical labels
    actual_labels = target_encoder.inverse_transform(y_test)
    predicted_labels = target_encoder.inverse_transform(preds)

    # ---------------------------
    # 📋 Create Results Table
    # ---------------------------
    results_df = pd.DataFrame({
        "Actual": actual_labels,
        "Predicted": predicted_labels
    })

    print("\n===== PREDICTED vs ACTUAL (first 20 rows) =====")
    # print(results_df.head(20))

    # ---------------------------
    # 📊 Confusion Matrix
    # ---------------------------
    # from sklearn.metrics import confusion_matrix
    # import seaborn as sns
    # import matplotlib.pyplot as plt

    # cm = confusion_matrix(actual_labels, predicted_labels, labels=target_encoder.classes_)

    # plt.figure(figsize=(7, 5))
    # sns.heatmap(
    #     cm, annot=True, fmt="d", cmap="Blues",
    #     xticklabels=target_encoder.classes_,
    #     yticklabels=target_encoder.classes_
    # )
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.title("Confusion Matrix - Test Results Prediction")
    # plt.show()

    # ---------------------------
    # 📐 METRICS
    # ---------------------------
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="weighted", zero_division=0)
    recall = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    report = classification_report(
        y_test, preds, target_names=target_encoder.classes_
    )

    print("\n===== MODEL METRICS =====")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("\nClassification Report:\n", report)

    # ---------------------------
    # ✅ RETURN INCLUDING ACTUAL & PREDICTED
    # ---------------------------
    return {
        "total_test_samples": len(X_test),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,

        # NEW PARTS YOU REQUESTED
        "predictions_preview": results_df.head(50).to_dict(orient="records"),
        # "actual_full": actual_labels.tolist(),
        # "predicted_full": predicted_labels.tolist()
    }


# ============================================================
# 🚀 RUN TEST
# ============================================================
# train_task2_model()
# print(test_task2_model())
def predict_single_patient_by_name(patient_name):

    print("\n=== PREDICT SINGLE PATIENT TEST RESULT ===\n")

    df = pd.read_csv(DATA_PATH)

    if "Name" not in df.columns:
        return {"error": "'Name' column missing in dataset"}

    # Case-insensitive match for safety
    row = df[df["Name"].str.lower() == patient_name.lower()]

    if row.empty:
        return {"error": f"No patient found with name '{patient_name}'"}

    # Store all patient details before dropping columns
    patient_details = row.to_dict(orient="records")[0]

    # Remove useless columns
    useless_cols = [
        "Name", "Doctor", "Hospital", "Room Number",
        "Insurance Provider", "Admission Type"
    ]
    row = row.drop(columns=[c for c in useless_cols if c in row.columns])

    # Load model + encoders
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    target_encoder = joblib.load(TARGET_ENCODER_PATH)
    categorical_cols = joblib.load(CAT_COLS_PATH)

    # Convert dates
    date_cols = ["Date of Admission", "Discharge Date"]
    for col in date_cols:
        row[col] = pd.to_datetime(row[col], errors="coerce")
        row[col] = row[col].astype("int64") // 10**9

    # Remove target if present
    if "Test Results" in row.columns:
        row = row.drop(columns=["Test Results"])
    # probs = model.predict_proba(row)[0]
    # probs = [float(f"{p:.2f}") for p in probs]
    # # ---- HARDCODED NEAREST TO 0.32 ----
    # TARGET_VALUE = 0.32
    # best_index = min(range(len(probs)), key=lambda i: abs(probs[i] - TARGET_VALUE))

    # predicted_label = target_encoder.inverse_transform([best_index])[0]

    # prob_map = {
    #     target_encoder.classes_[i]: float(probs[i])
    #     for i in range(len(probs))
    # }

    # return {
    #     "patient_name": patient_name,
    #     "patient_details": patient_details,
    #     "predicted_test_result": predicted_label,
    #     "probability_scores": prob_map,
    #     "nearest_to_value": TARGET_VALUE,
    #     "chosen_probability": float(probs[best_index])
    # }

    # Predict numeric class
    pred_numeric = model.predict(row).flatten()[0]
    # print(pred_numeric)

    # Convert to actual label
    predicted_label = target_encoder.inverse_transform([pred_numeric])[0]

    # Class probability scores
    probs = model.predict_proba(row)[0]
    prob_map = {
        target_encoder.classes_[i]: float(probs[i])
        for i in range(len(probs))
    }

    return {
        "patient_name": patient_name,
        "patient_details": patient_details,
        "predicted_test_result": predicted_label,
        "probability_scores": prob_map
    }
# print(predict_single_patient_by_name('eLIZABeTH jaCkSOn'))