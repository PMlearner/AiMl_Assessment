from fastapi import FastAPI
import uvicorn

from app.eda import run_eda
from app.supervised import test_task2_model,predict_single_patient_by_name
from app.unsupervised_model import detect_task3_anomalies
from app.llm_reporter import run_task4_recommendation
from app.patient_details import get_all_patients

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Medical Assessment API Up & Running"}

# -------- Task 1 API --------
@app.get("/eda")
def task1_eda():
    return run_eda()

# -------- Task 2 API --------
@app.get("/predict")
def task2_model():
    return test_task2_model()

# -------- Task 3 API --------
@app.get("/anomalies")
def task3_anomalies():
    return detect_task3_anomalies()

# -------- Task 4 API --------
@app.get("/recommend")
def doctor_recommend(patient_name: str):
    return run_task4_recommendation(patient_name)

# -------- Patient List API --------
@app.get("/patients")
def patient_list():
    return get_all_patients()
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )