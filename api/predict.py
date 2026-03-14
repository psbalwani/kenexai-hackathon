from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import numpy as np
import shap

app = FastAPI(title="Insurance Claim Prediction API")

model    = joblib.load("models/claim_model.pkl")
explainer = joblib.load("models/shap_explainer.pkl")

with open("models/feature_list.json") as f:
    FEATURES = json.load(f)

class DriverInput(BaseModel):
    AGE: int
    GENDER: int
    DRIVING_EXPERIENCE: int
    EDUCATION: int
    INCOME: int
    CREDIT_SCORE: float
    VEHICLE_OWNERSHIP: int
    VEHICLE_YEAR: int
    MARRIED: int
    CHILDREN: int
    ANNUAL_MILEAGE: int
    VEHICLE_TYPE: int
    SPEEDING_VIOLATIONS: int
    DUIS: int
    PAST_ACCIDENTS: int

@app.post("/predict")
def predict(driver: DriverInput):
    data = [getattr(driver, f) for f in FEATURES if f not in ["risk_score", "high_mileage_flag"]]

    risk_score       = driver.PAST_ACCIDENTS * 2 + driver.SPEEDING_VIOLATIONS + driver.DUIS * 3
    high_mileage_flag = 1 if driver.ANNUAL_MILEAGE > 15000 else 0

    data += [risk_score, high_mileage_flag]
    arr   = np.array([data])

    prob       = model.predict_proba(arr)[0][1]
    prediction = int(prob >= 0.5)

    risk_tier = "High" if risk_score >= 6 else "Medium" if risk_score >= 3 else "Low"

    # SHAP explanation
    shap_vals = explainer.shap_values(arr)
    top_factors = sorted(
        zip(FEATURES, shap_vals[0]),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    return {
        "claim_prediction":  prediction,
        "claim_probability": round(float(prob), 4),
        "risk_tier":         risk_tier,
        "risk_score":        risk_score,
        "top_risk_factors":  [{"feature": f, "impact": round(float(v), 4)}
                               for f, v in top_factors]
    }

@app.get("/health")
def health():
    return {"status": "ok"}