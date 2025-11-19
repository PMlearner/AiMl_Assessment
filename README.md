Medical Assessment System - FastAPI + Streamlit

This project provides a complete medical analytics pipeline including:

Automated Exploratory Data Analysis (EDA)

ML model prediction using CatBoost

Unsupervised anomaly detection

LLM-powered doctor recommendations

Interactive Streamlit dashboard

Backend is powered by FastAPI, frontend is built using Streamlit, and dependency management uses uv (uv.lock).

🚀 1. Project Structure
project/
│── app/
│   ├── eda.py
│   ├── supervised.py
│   ├── unsupervised_model.py
│   ├── llm_reporter.py
│   ├── patient_details.py
│   ├── __init__.py
│
│── main.py                # FastAPI backend
│── streamlit_app.py       # Streamlit UI
│── healthcare_dataset.csv # Dataset used by backend
│── .env                   # Environment variables
│── uv.lock                # Locked dependencies
│── README.md

🔧 2. Environment Setup

This project requires the following entries inside .env:

DATA_PATH=./healthcare_dataset.csv
OPENAI_API_KEY=your_openai_api_key_here


Make sure the .env file is in the root folder.

📦 3. Install Dependencies

This project uses uv as the package manager.

To install all dependencies from uv.lock:

uv sync


If uv is not installed, install it:

pip install uv

▶️ 4. Run the FastAPI Backend

Start the backend API server:

uv run python main.py


OR directly with Uvicorn:

uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload


Backend runs at:

http://localhost:8000

🖥️ 5. Run the Streamlit UI

In a new terminal:

uv run streamlit run streamlit_app.py


Streamlit UI opens at:

http://localhost:8501

🔌 6. Available API Endpoints
Endpoint	Method	Description
/	GET	API status
/eda	GET	Runs full EDA → returns distributions & frequencies
/predict	GET	ML predictions + metrics + preview table
/anomalies	GET	Detects anomaly billing patterns
/recommend?patient_name=X	GET	LLM doctor recommendation
/patients	GET	List of all patients
📊 7. Streamlit Features

The UI provides:

✅ Task 1 – EDA

Interactive Matplotlib/Seaborn plots:

Age distribution

Billing amount distribution

Room number distribution

Medical condition frequency

Admission type frequency

Medication frequency

✅ Task 2 – Prediction

Accuracy, Precision, Recall, F1 Score

Actual vs Predicted preview (table)

✅ Task 3 – Anomaly Detection

Total records

Anomalies count

Anomaly %

Anomaly table preview

✅ Task 4 – Doctor Recommendation

Dropdown of patient names

Generates:

Predicted test result

LLM doctor note

❗ 8. Troubleshooting
❓ Streamlit can't find API?

Check if backend is running:

http://localhost:8000

❓ OPENAI_API_KEY missing?

Ensure .env contains:

OPENAI_API_KEY=your_key

❓ Dataset not found?

Make sure:

DATA_PATH=./healthcare_dataset.csv

📝 9. License

This project is for internal/educational use only.