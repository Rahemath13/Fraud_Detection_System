import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# =========================
# 1. Load Dataset
# =========================
data = pd.read_csv(
    r"C:\Users\Dell\Downloads\creditcard_2023.csv\creditcard_2023.csv",
    nrows=20000
)
print("Dataset Shape:", data.shape)
print("Columns:", data.columns)
print(len(data.columns) - 1)

# =========================
# 2. Target Column
# =========================
# In creditcard_2023 dataset target column is usually 'Class'
# If different, change here.
target_column = "Class"

X = data.drop(target_column, axis=1)
y = data[target_column]
# =========================
# 3. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 4. Scaling
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 5. Model
# =========================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

# =========================
# 6. Evaluation
# =========================
y_pred = model.predict(X_test_scaled)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 7. Save Model
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(BASE_DIR, "model")

os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "fraud_model.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

print("\n✅ Model and Scaler Saved Successfully!")
