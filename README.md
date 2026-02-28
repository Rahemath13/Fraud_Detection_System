data/creditcard.csv
💳 Real-Time Credit Card Fraud Detection System

An end-to-end Machine Learning system for detecting fraudulent credit card transactions using supervised classification and real-time API deployment.

This project demonstrates not only model training but also full ML system deployment including backend API development, frontend integration, batch processing, and robust input validation.

🚀 Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent a very small percentage of total transactions.

This project builds a production-style ML inference pipeline with:

Model training and preprocessing

REST API deployment using FastAPI

Interactive frontend using Streamlit

Single and batch prediction support

Real-time fraud probability output

🧠 Problem Statement

Design and deploy a scalable machine learning system capable of detecting fraudulent transactions in real-time while handling:

Imbalanced data

Feature validation

Batch processing

API integration

Production-level error handling

📊 Dataset

Public Credit Card Fraud Detection dataset

284,807 transactions

30 numerical features:

Time

Amount

V1–V28 (PCA-transformed features)

Target variable: Class (0 = Legitimate, 1 = Fraud)

⚠ Note: Only feature columns are used for prediction.

🛠 Tech Stack

Languages & Libraries

Python

Pandas

NumPy

Scikit-learn

Backend

FastAPI

Uvicorn

Frontend

Streamlit

Communication

REST API

JSON Serialization

Requests library

🏗 System Architecture
User Input
   ↓
Streamlit Frontend
   ↓
FastAPI Backend
   ↓
Preprocessing (StandardScaler)
   ↓
Trained ML Model
   ↓
Prediction + Fraud Probability
   ↓
Frontend Display

⚙ Features

✅ Single transaction fraud prediction

✅ Batch prediction via CSV upload

✅ Fraud probability score display

✅ Robust input validation

✅ Tab/comma separated manual input handling

✅ Timeout handling for large datasets

✅ Error handling for feature mismatch

✅ Production-style REST API

📌 API Endpoints
🔹 Single Prediction

POST /predict

Request:

{
  "features": [30 numeric feature values]
}


Response:

{
  "prediction": 0,
  "fraud_probability": 0.0345
}

🔹 Batch Prediction

POST /predict_batch

Request:

{
  "features": [
    [30 feature values],
    [30 feature values]
  ]
}


Response:

[
  {"prediction": 0, "fraud_probability": 0.02},
  {"prediction": 1, "fraud_probability": 0.91}
]

📈 Model Development

Applied StandardScaler for feature normalization

Trained classification model for fraud detection

Evaluated using:

Precision

Recall

F1-score

ROC-AUC

Focused on handling imbalanced dataset challenges

🚨 Engineering Challenges Solved

API port mismatch between frontend and backend

JSON payload structure mismatch in batch inference

Feature count validation errors

Decimal parsing issues in manual input

Tab-separated data handling

Large dataset timeout errors

Data type consistency between frontend and backend

🔧 Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/https://github.com/Rahemath13/fraud-detection-system.git
cd fraud-detection-system

2️⃣ Create Virtual Environment
python -m venv .venv


Activate:

Windows:

.venv\Scripts\activate


Mac/Linux:

source .venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Start Backend
uvicorn main:app --reload --port 9000

5️⃣ Start Frontend
streamlit run app.py

🧪 How to Use
Manual Prediction

Enter 30 feature values (comma or tab separated)

Click "Predict Manual Transaction"

Batch Prediction

Upload CSV containing:

Time

Amount

V1–V28

Click "Predict Fraud for All Rows"

📊 Evaluation Metrics

Due to severe class imbalance, accuracy is not sufficient.

Primary metrics:

Precision

Recall

F1-score


ROC-AUC
