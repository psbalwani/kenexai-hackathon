import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genai.copilot import answer_question

# ── Config ──
st.set_page_config(
    page_title="Vehicle Insurance Risk Platform",
    page_icon="🚗",
    layout="wide"
)

@st.cache_resource
def load_db():
    return duckdb.connect("insurance.db", read_only=True)

@st.cache_resource
def load_model():
    model     = joblib.load("models/claim_model.pkl")
    explainer = joblib.load("models/shap_explainer.pkl")

    model.set_params(device="cpu")

    with open("models/feature_list.json") as f:
        features = json.load(f)
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    return model, explainer, features, metrics

con = load_db()

# ── Sidebar navigation ──
st.sidebar.image("https://img.icons8.com/fluency/96/car-insurance.png", width=80)
st.sidebar.title("Insurance Risk Platform")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 Risk Analytics",
    "🤖 ML Prediction",
    "💬 AI Copilot"
])

# ══════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🚗 Vehicle Insurance Risk Analytics Platform")
    st.caption("KD&A-8 | Kenexai Hackathon 2k26 | CHARUSAT")

    # KPIs from Motor
    total_policies  = con.execute("SELECT COUNT(*) FROM gold_motor").fetchone()[0]
    total_claims    = con.execute("SELECT COUNT(*) FROM gold_motor WHERE had_claim=1").fetchone()[0]
    avg_premium     = con.execute("SELECT ROUND(AVG(PREMIUM),2) FROM gold_motor").fetchone()[0]
    avg_claim_paid  = con.execute("SELECT ROUND(AVG(CLAIM_PAID),2) FROM gold_motor WHERE had_claim=1").fetchone()[0]
    claim_rate      = round(total_claims / total_policies * 100, 2)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Policies",  f"{total_policies:,}")
    c2.metric("Total Claims",    f"{total_claims:,}")
    c3.metric("Claim Rate",      f"{claim_rate}%")
    c4.metric("Avg Premium",     f"${avg_premium:,}")
    c5.metric("Avg Claim Paid",  f"${avg_claim_paid:,}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        df_veh = con.execute("""
            SELECT TYPE_VEHICLE, COUNT(*) AS policies,
                   ROUND(SUM(CLAIM_PAID),0) AS total_claims_paid
            FROM gold_motor
            GROUP BY TYPE_VEHICLE
            ORDER BY policies DESC
            LIMIT 10
        """).fetchdf()
        fig = px.bar(df_veh, x="TYPE_VEHICLE", y="policies",
                     title="Policies by Vehicle Type", color="total_claims_paid",
                     color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        df_make = con.execute("""
            SELECT MAKE, COUNT(*) AS policies
            FROM gold_motor
            GROUP BY MAKE
            ORDER BY policies DESC
            LIMIT 10
        """).fetchdf()
        fig2 = px.pie(df_make, names="MAKE", values="policies",
                      title="Top 10 Vehicle Makes")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        df_yr = con.execute("""
            SELECT CAST(PROD_YEAR AS INT) AS year, COUNT(*) AS cnt
            FROM gold_motor
            WHERE PROD_YEAR IS NOT NULL
            GROUP BY year ORDER BY year
        """).fetchdf()
        fig3 = px.line(df_yr, x="year", y="cnt",
                       title="Policies by Vehicle Production Year")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        df_usage = con.execute("""
            SELECT USAGE, ROUND(AVG(PREMIUM),2) AS avg_premium,
                   ROUND(AVG(CLAIM_PAID),2) AS avg_claim
            FROM gold_motor GROUP BY USAGE
        """).fetchdf()
        fig4 = px.bar(df_usage, x="USAGE", y=["avg_premium","avg_claim"],
                      barmode="group", title="Avg Premium vs Claim by Usage")
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════
# PAGE 2 — RISK ANALYTICS
# ══════════════════════════════════════════
elif page == "📊 Risk Analytics":
    st.title("📊 Driver Risk Analytics")

    # KPIs from Claims
    total_drivers = con.execute("SELECT COUNT(*) FROM gold_claims").fetchone()[0]
    high_risk     = con.execute("SELECT COUNT(*) FROM gold_claims WHERE risk_tier=2").fetchone()[0]
    claim_rate    = con.execute("SELECT ROUND(AVG(CAST(OUTCOME AS FLOAT))*100,2) FROM gold_claims").fetchone()[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Drivers",   f"{total_drivers:,}")
    c2.metric("High Risk",       f"{high_risk:,}")
    c3.metric("Overall Claim %", f"{claim_rate}%")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        df_age = con.execute("""
            SELECT AGE,
                   COUNT(*) AS drivers,
                   ROUND(AVG(CAST(OUTCOME AS FLOAT))*100,1) AS claim_rate_pct
            FROM silver_claims
            GROUP BY AGE ORDER BY claim_rate_pct DESC
        """).fetchdf()
        fig = px.bar(df_age, x="AGE", y="claim_rate_pct",
                     title="Claim Rate by Age Group (%)",
                     color="claim_rate_pct", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        df_exp = con.execute("""
            SELECT DRIVING_EXPERIENCE,
                   ROUND(AVG(CAST(OUTCOME AS FLOAT))*100,1) AS claim_rate_pct
            FROM silver_claims
            GROUP BY DRIVING_EXPERIENCE
        """).fetchdf()
        fig2 = px.bar(df_exp, x="DRIVING_EXPERIENCE", y="claim_rate_pct",
                      title="Claim Rate by Driving Experience (%)",
                      color="claim_rate_pct", color_continuous_scale="Blues")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        df_risk = con.execute("""
            SELECT risk_tier, COUNT(*) AS count
            FROM gold_claims GROUP BY risk_tier
        """).fetchdf()
        df_risk["risk_label"] = df_risk["risk_tier"].map({0:"Low",1:"Medium",2:"High"})
        fig3 = px.pie(df_risk, names="risk_label", values="count",
                      title="Risk Tier Distribution",
                      color_discrete_map={"Low":"green","Medium":"orange","High":"red"})
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        df_fi = pd.read_csv("models/feature_importance.csv").head(10)
        fig4 = px.bar(df_fi, x="importance", y="feature", orientation="h",
                      title="Top 10 Risk Factors (SHAP)",
                      color="importance", color_continuous_scale="Oranges")
        st.plotly_chart(fig4, use_container_width=True)

    # Postal code heatmap
    st.subheader("📍 Risk by Postal Code")
    df_postal = con.execute("""
        SELECT POSTAL_CODE,
               COUNT(*) AS drivers,
               ROUND(AVG(CAST(OUTCOME AS FLOAT))*100,1) AS claim_rate_pct
        FROM silver_claims
        GROUP BY POSTAL_CODE
        ORDER BY claim_rate_pct DESC
        LIMIT 20
    """).fetchdf()
    fig5 = px.bar(df_postal, x="POSTAL_CODE", y="claim_rate_pct",
                  title="Claim Rate by Postal Code (Top 20)",
                  color="claim_rate_pct", color_continuous_scale="Reds")
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════
# PAGE 3 — ML PREDICTION
# ══════════════════════════════════════════
elif page == "🤖 ML Prediction":
    st.title("🤖 Claim Risk Predictor")

    try:
        model, explainer, features, metrics = load_model()
        st.success(f"Model loaded — Accuracy: {metrics['accuracy']*100:.1f}% | ROC-AUC: {metrics['roc_auc']:.3f}")
    except:
        st.error("Run models/train_model.py first")
        st.stop()

    st.subheader("Enter Driver Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age_map = {"16-25":0, "26-39":1, "40-64":2, "65+":3}
        age_sel = st.selectbox("Age Group", list(age_map.keys()))

        gen_map = {"Female":0, "Male":1}
        gen_sel = st.selectbox("Gender", list(gen_map.keys()))

        exp_map = {"0-9y":0, "10-19y":1, "20-29y":2, "30y+":3}
        exp_sel = st.selectbox("Driving Experience", list(exp_map.keys()))

        edu_map = {"High School":0, "None":1, "University":2}
        edu_sel = st.selectbox("Education", list(edu_map.keys()))

        inc_map = {"poverty":0, "working class":1, "middle class":2, "upper class":3}
        inc_sel = st.selectbox("Income Class", list(inc_map.keys()))

    with col2:
        credit_score    = st.slider("Credit Score", 0.0, 1.0, 0.5, 0.01)
        vehicle_own     = st.selectbox("Owns Vehicle", {0:"No", 1:"Yes"}.keys())
        veh_year_map    = {"after 2015":0, "before 2015":1}
        veh_year_sel    = st.selectbox("Vehicle Year", list(veh_year_map.keys()))
        married         = st.selectbox("Married", {0:"No", 1:"Yes"}.keys())
        children        = st.selectbox("Has Children", {0:"No", 1:"Yes"}.keys())

    with col3:
        annual_mileage  = st.slider("Annual Mileage", 5000, 30000, 12000, 500)
        veh_type_map    = {"sedan":0, "pickup":1, "suv":2}
        veh_type_sel    = st.selectbox("Vehicle Type", list(veh_type_map.keys()))
        speeding        = st.number_input("Speeding Violations", 0, 20, 0)
        duis            = st.number_input("DUIs", 0, 10, 0)
        accidents       = st.number_input("Past Accidents", 0, 20, 0)

    if st.button("🔍 Predict Claim Risk", type="primary"):
        risk_score        = int(accidents) * 2 + int(speeding) + int(duis) * 3
        high_mileage_flag = 1 if annual_mileage > 15000 else 0

        input_data = np.array([[
            age_map[age_sel], gen_map[gen_sel], exp_map[exp_sel],
            edu_map[edu_sel], inc_map[inc_sel], credit_score,
            int(vehicle_own == "Yes"), veh_year_map[veh_year_sel],
            int(married == "Yes"), int(children == "Yes"),
            annual_mileage, veh_type_map[veh_type_sel],
            int(speeding), int(duis), int(accidents),
            risk_score, high_mileage_flag
        ]])

        prob       = model.predict_proba(input_data)[0][1]
        prediction = int(prob >= 0.5)
        risk_tier  = "🔴 HIGH" if risk_score >= 6 else "🟡 MEDIUM" if risk_score >= 3 else "🟢 LOW"

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Claim Prediction",  "⚠️ CLAIM" if prediction else "✅ NO CLAIM")
        r2.metric("Claim Probability", f"{prob*100:.1f}%")
        r3.metric("Risk Tier",         risk_tier)

        # Suggested premium
        base_premium = 400
        suggested    = base_premium + risk_score * 100 + int(accidents) * 150
        st.info(f"💰 **Suggested Premium:** ${suggested}/year  |  Risk Score: {risk_score}")

        # SHAP explanation
        shap_vals   = explainer.shap_values(input_data)
        shap_df     = pd.DataFrame({
            "feature": features,
            "impact":  shap_vals[0]
        }).sort_values("impact", key=abs, ascending=False).head(5)

        st.subheader("🧠 Why this prediction? (AI Explanation)")
        for _, row in shap_df.iterrows():
            direction = "⬆️ increases" if row["impact"] > 0 else "⬇️ reduces"
            st.write(f"**{row['feature']}** {direction} claim risk (impact: {row['impact']:.3f})")


# ══════════════════════════════════════════
# PAGE 4 — GENAI COPILOT
# ══════════════════════════════════════════
elif page == "💬 AI Copilot":
    st.title("💬 Insurance AI Copilot")
    st.caption("Powered by GPT — answers based on your actual data only")

    from genai.copilot import answer_question

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Suggested questions
    if not st.session_state.chat_history:
        st.subheader("Try asking:")
        q_cols = st.columns(3)
        questions = [
            "Which age group has the highest claim rate?",
            "What vehicle type has the most claims?",
            "What is the average premium by vehicle usage?"
        ]
        for i, q in enumerate(questions):
            if q_cols[i].button(q):
                st.session_state.chat_history.append({"role":"user","content":q})
                with st.spinner("Thinking..."):
                    answer = answer_question(q)
                st.session_state.chat_history.append({"role":"assistant","content":answer})
                st.rerun()

    # Chat input
    user_input = st.chat_input("Ask anything about the insurance data...")
    if user_input:
        st.session_state.chat_history.append({"role":"user","content":user_input})
        with st.spinner("Analyzing data..."):
            answer = answer_question(user_input)
        st.session_state.chat_history.append({"role":"assistant","content":answer})
        st.rerun()