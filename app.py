"""try-self — Streamlit demo UI for loan-approval prediction.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src.predict import load_model, predict_application

st.set_page_config(page_title="try-self · Loan Approval", page_icon="🏦", layout="centered")

st.title("🏦 try-self — Loan Approval Prediction")
st.caption(
    "Fill in the application and get an instant decision with an explanation. "
    "Educational project — not a real underwriting system."
)

try:
    load_model()
except FileNotFoundError:
    st.error("No trained model found. Run `python -m src.train` first, then refresh.")
    st.stop()

with st.form("application"):
    c1, c2 = st.columns(2)
    with c1:
        applicant_income = st.number_input("Applicant monthly income ($)", 0, 1_000_000, 5400, step=100)
        coapplicant_income = st.number_input("Co-applicant monthly income ($)", 0, 1_000_000, 1200, step=100)
        loan_amount = st.number_input("Loan amount ($)", 1000, 10_000_000, 128_000, step=1000)
        loan_term_months = st.selectbox("Loan term (months)", [120, 180, 240, 300, 360], index=4)
    with c2:
        credit_history = st.selectbox("Credit history", [1, 0], format_func=lambda v: "Good (1)" if v else "Bad / none (0)")
        dependents = st.number_input("Dependents", 0, 10, 0)
        employment_status = st.selectbox("Employment status", ["salaried", "self_employed", "unemployed"])
        property_area = st.selectbox("Property area", ["urban", "semiurban", "rural"])
    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    result = predict_application(
        {
            "applicant_income": applicant_income,
            "coapplicant_income": coapplicant_income,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term_months,
            "credit_history": credit_history,
            "dependents": dependents,
            "employment_status": employment_status,
            "property_area": property_area,
        }
    )
    approved = result["decision"] == "Approved"
    (st.success if approved else st.error)(
        f"**{result['decision']}** — approval probability {result['probability']:.0%}"
    )
    st.progress(result["probability"])

    st.subheader("Why?")
    st.caption("Top factors pushing this decision (positive = toward approval).")
    for f in result["top_factors"]:
        arrow = "🟢 ▲" if f["impact"] > 0 else "🔴 ▼"
        st.write(f"{arrow} `{f['feature']}` — impact {f['impact']:+.3f}")
