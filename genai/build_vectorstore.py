import duckdb
import os
import pickle
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv()

con = duckdb.connect("insurance.db", read_only=True)
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []

# Motor chunks
motor_chunks = con.execute("""
    SELECT TYPE_VEHICLE, USAGE,
        COUNT(*) AS total_policies,
        ROUND(AVG(PREMIUM), 2) AS avg_premium,
        ROUND(AVG(INSURED_VALUE), 2) AS avg_insured_value,
        ROUND(AVG(CLAIM_PAID), 2) AS avg_claim_paid,
        ROUND(AVG(had_claim)*100, 1) AS claim_rate_pct,
        ROUND(MIN(PREMIUM), 2) AS min_premium,
        ROUND(MAX(PREMIUM), 2) AS max_premium
    FROM gold_motor GROUP BY TYPE_VEHICLE, USAGE
""").fetchdf()

for _, row in motor_chunks.iterrows():
    documents.append({
        "text": f"Vehicle Type: {row['TYPE_VEHICLE']}, Usage: {row['USAGE']}. "
                f"Total policies: {row['total_policies']}. "
                f"Average premium: ${row['avg_premium']}. "
                f"Premium range: ${row['min_premium']} to ${row['max_premium']}. "
                f"Average insured value: ${row['avg_insured_value']}. "
                f"Average claim paid: ${row['avg_claim_paid']}. "
                f"Claim rate: {row['claim_rate_pct']}%.",
        "source": "motor"
    })

# Make chunks
make_chunks = con.execute("""
    SELECT MAKE, TYPE_VEHICLE,
        COUNT(*) AS total_policies,
        ROUND(AVG(PREMIUM), 2) AS avg_premium,
        ROUND(AVG(CLAIM_PAID), 2) AS avg_claim_paid,
        ROUND(AVG(had_claim)*100, 1) AS claim_rate_pct
    FROM gold_motor WHERE MAKE IS NOT NULL
    GROUP BY MAKE, TYPE_VEHICLE
    ORDER BY total_policies DESC LIMIT 60
""").fetchdf()

for _, row in make_chunks.iterrows():
    documents.append({
        "text": f"Vehicle make: {row['MAKE']}, type: {row['TYPE_VEHICLE']}. "
                f"Total policies: {row['total_policies']}. "
                f"Average premium: ${row['avg_premium']}. "
                f"Average claim paid: ${row['avg_claim_paid']}. "
                f"Claim rate: {row['claim_rate_pct']}%.",
        "source": "motor_make"
    })

# Claims chunks
claims_chunks = con.execute("""
    SELECT AGE, DRIVING_EXPERIENCE,
        COUNT(*) AS total_drivers,
        ROUND(AVG(CAST(OUTCOME AS FLOAT))*100, 1) AS claim_rate_pct,
        ROUND(AVG(ANNUAL_MILEAGE), 0) AS avg_mileage,
        ROUND(AVG(CREDIT_SCORE), 3) AS avg_credit_score,
        ROUND(AVG(PAST_ACCIDENTS), 2) AS avg_accidents
    FROM silver_claims GROUP BY AGE, DRIVING_EXPERIENCE
""").fetchdf()

for _, row in claims_chunks.iterrows():
    documents.append({
        "text": f"Driver age: {row['AGE']}, experience: {row['DRIVING_EXPERIENCE']}. "
                f"Total drivers: {row['total_drivers']}. "
                f"Claim rate: {row['claim_rate_pct']}%. "
                f"Avg mileage: {row['avg_mileage']}. "
                f"Avg credit score: {row['avg_credit_score']}. "
                f"Avg past accidents: {row['avg_accidents']}.",
        "source": "claims_age"
    })

# Risk tier chunks
risk_chunks = con.execute("""
    SELECT CASE risk_tier WHEN 0 THEN 'Low' WHEN 1 THEN 'Medium' ELSE 'High' END AS risk_level,
        COUNT(*) AS total_drivers,
        ROUND(AVG(CAST(OUTCOME AS FLOAT))*100, 1) AS claim_rate_pct,
        ROUND(AVG(risk_score), 2) AS avg_risk_score
    FROM gold_claims GROUP BY risk_tier
""").fetchdf()

for _, row in risk_chunks.iterrows():
    documents.append({
        "text": f"Risk tier: {row['risk_level']}. "
                f"Total drivers: {row['total_drivers']}. "
                f"Claim rate: {row['claim_rate_pct']}%. "
                f"Average risk score: {row['avg_risk_score']}.",
        "source": "claims_risk"
    })

print(f"✅ Total chunks: {len(documents)}")

# Embed
texts = [d["text"] for d in documents]
print("⏳ Embedding chunks...")
embeddings = model.encode(texts, show_progress_bar=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

# Save
os.makedirs("genai/vectorstore", exist_ok=True)
faiss.write_index(index, "genai/vectorstore/index.faiss")
with open("genai/vectorstore/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print(f"✅ FAISS vectorstore saved — {len(documents)} vectors indexed")