import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
def get_all_patients():
    df = pd.read_csv(os.getenv("data_path"))
    return df["Name"].dropna().tolist()
# print(get_all_patients())