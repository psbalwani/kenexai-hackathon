import duckdb
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import shap

# ── Load from Gold layer ──
con = duckdb.connect("insurance.db")
df  = con.execute("SELECT * FROM gold_claims").fetchdf()

# ── Features & Target ──
FEATURES = [
    "AGE", "GENDER", "DRIVING_EXPERIENCE", "EDUCATION",
    "INCOME", "CREDIT_SCORE", "VEHICLE_OWNERSHIP", "VEHICLE_YEAR",
    "MARRIED", "CHILDREN", "ANNUAL_MILEAGE", "VEHICLE_TYPE",
    "SPEEDING_VIOLATIONS", "DUIS", "PAST_ACCIDENTS",
    "risk_score", "high_mileage_flag"
]
TARGET = "OUTCOME"

X = df[FEATURES]
y = df[TARGET]

# ── Train / Test Split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train Model ──
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# ── Evaluate ──
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy : {acc:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print(classification_report(y_test, y_pred))

# ── SHAP Feature Importance ──
explainer    = shap.TreeExplainer(model)
shap_values  = explainer.shap_values(X_test)
feature_importance = pd.DataFrame({
    "feature":    FEATURES,
    "importance": np.abs(shap_values[1]).mean(axis=0)
}).sort_values("importance", ascending=False)

print("\nTop Features:")
print(feature_importance.head(10))

# ── Save Everything ──
joblib.dump(model, "models/claim_model.pkl")
joblib.dump(explainer, "models/shap_explainer.pkl")

feature_importance.to_csv("models/feature_importance.csv", index=False)

metrics = {"accuracy": round(acc, 4), "roc_auc": round(auc, 4)}
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f)

with open("models/feature_list.json", "w") as f:
    json.dump(FEATURES, f)

print("Model, explainer and metrics saved to models/")