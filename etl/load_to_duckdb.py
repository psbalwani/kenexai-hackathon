import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

con = duckdb.connect("insurance.db")

# BRONZE — raw ingestion
claims_raw = pd.read_csv("data/raw/Car_Insurance_Claim.csv")
motor_raw  = pd.read_csv("data/raw/motor_data14-2018.csv")

con.execute("DROP TABLE IF EXISTS bronze_claims")
con.execute("DROP TABLE IF EXISTS bronze_motor")
con.execute("CREATE TABLE bronze_claims AS SELECT * FROM claims_raw")
con.execute("CREATE TABLE bronze_motor  AS SELECT * FROM motor_raw")
print("Bronze loaded")

# SILVER — cleaned + typed
claims = claims_raw.drop_duplicates().dropna()
motor  = motor_raw.drop_duplicates()

motor["CLAIM_PAID"]    = pd.to_numeric(motor["CLAIM_PAID"],    errors="coerce").fillna(0)
motor["INSURED_VALUE"] = pd.to_numeric(motor["INSURED_VALUE"], errors="coerce")
motor["PREMIUM"]       = pd.to_numeric(motor["PREMIUM"],       errors="coerce")
motor["PROD_YEAR"]     = pd.to_numeric(motor["PROD_YEAR"],     errors="coerce")
motor = motor.dropna(subset=["INSURED_VALUE", "PREMIUM"])

motor["INSR_BEGIN"] = pd.to_datetime(motor["INSR_BEGIN"], format="%d-%b-%y", errors="coerce")
motor["INSR_END"]   = pd.to_datetime(motor["INSR_END"],   format="%d-%b-%y", errors="coerce")

con.execute("DROP TABLE IF EXISTS silver_claims")
con.execute("DROP TABLE IF EXISTS silver_motor")
con.execute("CREATE TABLE silver_claims AS SELECT * FROM claims")
con.execute("CREATE TABLE silver_motor  AS SELECT * FROM motor")
print("Silver loaded")

# GOLD — feature engineered
# Gold Claims — for ML
le = LabelEncoder()
gold_claims = claims.copy()
cat_cols = ["AGE", "GENDER", "RACE", "DRIVING_EXPERIENCE",
            "EDUCATION", "INCOME", "VEHICLE_YEAR", "VEHICLE_TYPE"]
for col in cat_cols:
    gold_claims[col] = le.fit_transform(gold_claims[col].astype(str))

gold_claims["risk_score"] = (
    gold_claims["PAST_ACCIDENTS"] * 2 +
    gold_claims["SPEEDING_VIOLATIONS"] +
    gold_claims["DUIS"] * 3
)
gold_claims["high_mileage_flag"] = (gold_claims["ANNUAL_MILEAGE"] > 15000).astype(int)
gold_claims["risk_tier"] = pd.cut(
    gold_claims["risk_score"],
    bins=[-1, 2, 5, 100],
    labels=[0, 1, 2]  # 0=Low, 1=Medium, 2=High
).astype(int)

# Gold Motor — for analytics
gold_motor = motor.copy()
gold_motor["had_claim"]            = (gold_motor["CLAIM_PAID"] > 0).astype(int)
gold_motor["premium_rate_pct"]     = (gold_motor["PREMIUM"] / gold_motor["INSURED_VALUE"].replace(0, np.nan) * 100).round(4)
gold_motor["policy_duration_days"] = (gold_motor["INSR_END"] - gold_motor["INSR_BEGIN"]).dt.days
gold_motor["vehicle_age"]          = 2024 - gold_motor["PROD_YEAR"].fillna(2010)

con.execute("DROP TABLE IF EXISTS gold_claims")
con.execute("DROP TABLE IF EXISTS gold_motor")
con.execute("CREATE TABLE gold_claims AS SELECT * FROM gold_claims")
con.execute("CREATE TABLE gold_motor  AS SELECT * FROM gold_motor")

print("Gold loaded")
print("  gold_claims rows:", con.execute("SELECT COUNT(*) FROM gold_claims").fetchone()[0])
print("  gold_motor  rows:", con.execute("SELECT COUNT(*) FROM gold_motor").fetchone()[0])