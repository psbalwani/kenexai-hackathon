import pandas as pd
import numpy as np
import os

def generate_new_claims(n=50):
    """Simulates n new insurance claim records arriving"""
    
    ages = ["16-25", "26-39", "40-64", "65+"]
    genders = ["Male", "Female"]
    races = ["majority", "minority"]
    experiences = ["0-9y", "10-19y", "20-29y", "30y+"]
    educations = ["high school", "none", "university"]
    incomes = ["poverty", "working class", "middle class", "upper class"]
    vehicle_years = ["after 2015", "before 2015"]
    vehicle_types = ["sedan", "pickup"]

    new_data = pd.DataFrame({
        "ID"                  : range(100000, 100000 + n),
        "AGE"                 : np.random.choice(ages, n),
        "GENDER"              : np.random.choice(genders, n),
        "RACE"                : np.random.choice(races, n),
        "DRIVING_EXPERIENCE"  : np.random.choice(experiences, n),
        "EDUCATION"           : np.random.choice(educations, n),
        "INCOME"              : np.random.choice(incomes, n),
        "CREDIT_SCORE"        : np.round(np.random.uniform(0.1, 1.0, n), 3),
        "VEHICLE_OWNERSHIP"   : np.random.choice([0, 1], n),
        "VEHICLE_YEAR"        : np.random.choice(vehicle_years, n),
        "MARRIED"             : np.random.choice([0, 1], n),
        "CHILDREN"            : np.random.choice([0, 1], n),
        "POSTAL_CODE"         : np.random.choice([10238, 32765, 92101], n),
        "ANNUAL_MILEAGE"      : np.random.randint(5000, 30000, n),
        "VEHICLE_TYPE"        : np.random.choice(vehicle_types, n),
        "SPEEDING_VIOLATIONS" : np.random.randint(0, 6, n),
        "DUIS"                : np.random.randint(0, 3, n),
        "PAST_ACCIDENTS"      : np.random.randint(0, 5, n),
        "OUTCOME"             : np.random.choice([0, 1], n, p=[0.7, 0.3])
    })
    return new_data

def simulate_ingestion():
    path = "data/raw/Car_Insurance_Claim.csv"
    existing = pd.read_csv(path)
    new_records = generate_new_claims(50)
    
    # Append new records
    updated = pd.concat([existing, new_records], ignore_index=True)
    updated.to_csv(path, index=False)
    
    print(f"✅ Ingested 50 new records — total now: {len(updated)}")

if __name__ == "__main__":
    simulate_ingestion()