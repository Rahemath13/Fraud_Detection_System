from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# -------------------------
# Load Model Safely
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model", "fraud_model.pkl")
scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# -------------------------
# Input Schema
# -------------------------
class Transaction(BaseModel):
    features: list

# ✅ NEW SCHEMA FOR BATCH
class BatchTransaction(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def predict(transaction: Transaction):
    print("Prediction request received")

    features = np.array(transaction.features).reshape(1, -1)
    scaled = scaler.transform(features)

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability)
    }

# 🚀 NEW FAST BATCH ENDPOINT
@app.post("/predict_batch")
def predict_batch(transactions: BatchTransaction):

    features = np.array(transactions.features)

    scaled = scaler.transform(features)

    predictions = model.predict(scaled)
    probabilities = model.predict_proba(scaled)[:, 1]

    results = []

    for pred, prob in zip(predictions, probabilities):
        results.append({
            "prediction": int(pred),
            "fraud_probability": float(prob)
        })

    return results
from fastapi import UploadFile, File
import pandas as pd

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        print("CSV Prediction request received")

        df = pd.read_csv(file.file)

        # Ensure correct feature order (VERY IMPORTANT)
        df = df[scaler.feature_names_in_]

        scaled = scaler.transform(df)

        predictions = model.predict(scaled)
        probabilities = model.predict_proba(scaled)[:, 1]

        results = []

        for pred, prob in zip(predictions, probabilities):
            results.append({
                "prediction": int(pred),
                "fraud_probability": float(prob)
            })

        return results

    except Exception as e:
        return {"error": str(e)}
