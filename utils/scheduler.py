import schedule
import time
import subprocess
from utils.data_simulator import simulate_ingestion

def run_pipeline():
    print("⏰ Scheduled pipeline running...")
    simulate_ingestion()                              # ← new data arrives
    subprocess.run(["python", "etl/load_to_duckdb.py"])   # ← ETL processes it
    subprocess.run(["python", "genai/build_vectorstore.py"]) # ← RAG refreshes
    print("✅ Pipeline complete")

schedule.every(5).minutes.do(run_pipeline)

print("🚀 Pipeline scheduler started — runs every 5 minutes")
while True:
    schedule.run_pending()
    time.sleep(1)