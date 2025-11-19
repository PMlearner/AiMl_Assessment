import pandas as pd
from openai import OpenAI
from app.supervised import predict_single_patient_by_name  # <-- Your Task 2 function
from dotenv import load_dotenv
load_dotenv()
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_task4_recommendation(patient_name):

    # 1️⃣ Get prediction from Task-2
    result = predict_single_patient_by_name(patient_name)

    if "error" in result:
        return result   # Return error if patient not found

    predicted_label = result["predicted_test_result"]
    patient_details = result["patient_details"]

    # Build prompt
    prompt = f"""
You are a senior medical doctor. Create a concise, professional medical recommendation.

Patient Details:
- Name: {patient_details.get('Name')}
- Age: {patient_details.get('Age')}
- Medical Condition: {patient_details.get('Medical Condition')}
- Medication: {patient_details.get('Medication')}
- Predicted Test Result: {predicted_label}

Generate a short doctor-style recommendation as a summary report.
Include health advice based on the predicted result.

Tone must be: professional, simple, and medically accurate.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    doctor_note = response.choices[0].message.content

    return {
        "patient_name": patient_name,
        "predicted_test_result": predicted_label,
        "recommendation": doctor_note
    }
# print(run_task4_recommendation('eLIZABeTH jaCkSOn'))