"""
try-self — Streamlit UI for Loan Approval Prediction

Run:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.predict import load_model, predict_application


# ===================================================
# PAGE CONFIG
# ===================================================

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===================================================
# CUSTOM CSS
# ===================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 0px;
        }

        .subtitle {
            font-size: 18px;
            color: #9ca3af;
            margin-bottom: 25px;
        }

        .footer {
            text-align: center;
            color: gray;
            padding-top: 40px;
            font-size: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ===================================================
# EMI CALCULATION
# ===================================================

def calculate_emi(principal, annual_rate, months):
    """
    Calculate monthly EMI.

    principal   : Loan amount
    annual_rate : Annual interest rate in %
    months      : Loan duration in months
    """

    monthly_rate = annual_rate / 12 / 100

    if monthly_rate == 0:
        return principal / months

    emi = (
        principal
        * monthly_rate
        * (1 + monthly_rate) ** months
        / ((1 + monthly_rate) ** months - 1)
    )

    return emi


# ===================================================
# REJECTION REASON FUNCTION
# ===================================================

def get_rejection_reasons(
    credit_score,
    applicant_income,
    coapplicant_income,
    loan_amount,
    emi,
    employment_status,
    dependents,
    top_factors
):
    """
    Generate understandable possible reasons
    for loan rejection.
    """

    reasons = []

    total_income = applicant_income + coapplicant_income

    # ------------------------------------------------
    # MODEL NEGATIVE FACTORS
    # ------------------------------------------------

    for factor in top_factors:

        if factor.get("impact", 0) < 0:

            feature = (
                factor.get("feature", "Unknown Factor")
                .replace("_", " ")
                .title()
            )

            reason = (
                f"{feature} may have negatively affected "
                f"the approval decision."
            )

            if reason not in reasons:
                reasons.append(reason)

    # ------------------------------------------------
    # CREDIT SCORE
    # ------------------------------------------------

    if credit_score < 600:

        reasons.append(
            "Your credit score is below 600, which may indicate "
            "a higher lending risk."
        )

    # ------------------------------------------------
    # EMI TO INCOME RATIO
    # ------------------------------------------------

    if total_income > 0:

        emi_ratio = emi / total_income

        if emi_ratio > 0.50:

            reasons.append(
                "The estimated monthly EMI is more than 50% "
                "of the total monthly household income."
            )

        elif emi_ratio > 0.40:

            reasons.append(
                "The estimated monthly EMI is relatively high "
                "compared to the total monthly household income."
            )

    else:

        reasons.append(
            "No monthly household income was provided."
        )

    # ------------------------------------------------
    # EMPLOYMENT STATUS
    # ------------------------------------------------

    if employment_status == "unemployed":

        reasons.append(
            "Unemployment may reduce the ability to repay "
            "the loan."
        )

    # ------------------------------------------------
    # HIGH LOAN COMPARED TO ANNUAL INCOME
    # ------------------------------------------------

    annual_income = total_income * 12

    if annual_income > 0:

        loan_income_ratio = loan_amount / annual_income

        if loan_income_ratio > 5:

            reasons.append(
                "The requested loan amount is very high "
                "compared to the annual household income."
            )

    # ------------------------------------------------
    # DEPENDENTS
    # ------------------------------------------------

    if dependents >= 5:

        reasons.append(
            "A high number of dependents may increase "
            "monthly financial responsibilities."
        )

    # ------------------------------------------------
    # FALLBACK
    # ------------------------------------------------

    if not reasons:

        reasons.append(
            "The machine learning model predicted an approval "
            "probability below its decision threshold based on "
            "the combination of application features."
        )

    return reasons


# ===================================================
# SESSION STATE
# ===================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ===================================================
# SIDEBAR
# ===================================================

with st.sidebar:

    st.title("🏦 Loan Predictor")

    st.markdown("---")

    st.subheader("About")

    st.write(
        """
        This application uses a Machine Learning model
        to estimate the probability of loan approval.
        """
    )

    st.markdown("---")

    st.subheader("How It Works")

    st.write("1️⃣ Enter applicant details")
    st.write("2️⃣ Enter loan details")
    st.write("3️⃣ Select interest rate")
    st.write("4️⃣ Click Predict")
    st.write("5️⃣ View prediction, EMI and insights")

    st.markdown("---")

    if st.button(
        "🗑 Clear Prediction History",
        use_container_width=True
    ):
        st.session_state.history = []
        st.rerun()

    st.markdown("---")

    st.caption(
        "⚠️ Educational project only. "
        "This application should not be used for actual "
        "financial or lending decisions."
    )


# ===================================================
# HEADER
# ===================================================

st.markdown(
    '<div class="main-title">🏦 Loan Approval Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Enter applicant and loan details to get a machine-learning
        based loan prediction along with EMI information.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ===================================================
# LOAD ML MODEL
# ===================================================

try:
    load_model()

except FileNotFoundError:

    st.error(
        "❌ No trained model found. "
        "Run `python -m src.train` first, then refresh."
    )

    st.stop()


# ===================================================
# APPLICATION FORM
# ===================================================

st.subheader("📋 Loan Application Details")


with st.form("application"):

    col1, col2 = st.columns(2, gap="large")


    # -----------------------------------------------
    # APPLICANT DETAILS
    # -----------------------------------------------

    with col1:

        st.subheader("👤 Applicant Information")

        applicant_income = st.number_input(
            "Applicant Monthly Income (₹)",
            min_value=0,
            max_value=10_000_000,
            value=50_000,
            step=1_000
        )

        coapplicant_income = st.number_input(
            "Co-applicant Monthly Income (₹)",
            min_value=0,
            max_value=10_000_000,
            value=20_000,
            step=1_000
        )

        credit_score = st.slider(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=700,
            step=10
        )

        dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

        employment_status = st.selectbox(
            "Employment Status",
            [
                "salaried",
                "self_employed",
                "unemployed"
            ]
        )


    # -----------------------------------------------
    # LOAN DETAILS
    # -----------------------------------------------

    with col2:

        st.subheader("💰 Loan Information")

        loan_amount = st.number_input(
            "Loan Amount (₹)",
            min_value=10_000,
            max_value=100_000_000,
            value=1_000_000,
            step=10_000
        )

        loan_term_years = st.selectbox(
            "Loan Term (Years)",
            [5, 10, 15, 20, 25, 30],
            index=5
        )

        interest_rate = st.slider(
            "Annual Interest Rate (%)",
            min_value=0.0,
            max_value=25.0,
            value=8.5,
            step=0.1
        )

        property_area = st.selectbox(
            "Property Area",
            [
                "urban",
                "semiurban",
                "rural"
            ]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "🔍 Predict Loan Approval",
        use_container_width=True
    )


# ===================================================
# PREDICTION
# ===================================================

if submitted:

    # -----------------------------------------------
    # CONVERT YEARS TO MONTHS
    # -----------------------------------------------

    loan_term_months = loan_term_years * 12


    # -----------------------------------------------
    # CREDIT SCORE CONVERSION
    # -----------------------------------------------

    # Existing ML model expects credit_history.
    # Score >= 600 is considered good.

    credit_history = 1 if credit_score >= 600 else 0


    # -----------------------------------------------
    # INPUT DATA FOR ML MODEL
    # -----------------------------------------------

    input_data = {

        "applicant_income": applicant_income,

        "coapplicant_income": coapplicant_income,

        "loan_amount": loan_amount,

        "loan_term_months": loan_term_months,

        "credit_history": credit_history,

        "dependents": dependents,

        "employment_status": employment_status,

        "property_area": property_area,
    }


    # -----------------------------------------------
    # ML PREDICTION
    # -----------------------------------------------

    result = predict_application(input_data)

    approved = result["decision"] == "Approved"

    probability = result["probability"]


    # ===================================================
    # CALCULATE EMI FOR EVERY APPLICATION
    # ===================================================

    emi = calculate_emi(
        principal=loan_amount,
        annual_rate=interest_rate,
        months=loan_term_months
    )


    # ===================================================
    # RESULT SECTION
    # ===================================================

    st.divider()

    st.subheader("📊 Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)


    # -----------------------------------------------
    # DECISION
    # -----------------------------------------------

    with result_col1:

        if approved:

            st.success("## ✅ Approved")

        else:

            st.error("## ❌ Rejected")

        st.metric(
            "Approval Probability",
            f"{probability:.1%}"
        )


    # -----------------------------------------------
    # EMI - ALWAYS SHOWN
    # -----------------------------------------------

    with result_col2:

        st.metric(
            "Estimated Monthly EMI",
            f"₹{emi:,.2f}"
        )

        st.caption(
            f"Based on {interest_rate:.1f}% annual interest"
        )


    # -----------------------------------------------
    # CREDIT SCORE
    # -----------------------------------------------

    with result_col3:

        st.metric(
            "Credit Score",
            credit_score
        )

        if credit_score >= 750:

            st.success("Excellent Credit Score")

        elif credit_score >= 600:

            st.info("Good Credit Score")

        else:

            st.warning("Credit Score Needs Improvement")


    # ===================================================
    # APPROVAL PROBABILITY
    # ===================================================

    st.subheader("📈 Approval Probability")

    st.progress(probability)

    st.caption(
        f"The ML model estimates a {probability:.1%} "
        "probability of loan approval."
    )


    # ===================================================
    # REJECTION REASONS
    # ===================================================

    if not approved:

        st.divider()

        st.subheader("❓ Possible Reasons for Rejection")

        reasons = get_rejection_reasons(
            credit_score=credit_score,
            applicant_income=applicant_income,
            coapplicant_income=coapplicant_income,
            loan_amount=loan_amount,
            emi=emi,
            employment_status=employment_status,
            dependents=dependents,
            top_factors=result.get("top_factors", [])
        )

        st.warning(
            "The following factors may have contributed "
            "to the rejection:"
        )

        for reason in reasons:

            st.write(f"🔴 {reason}")


    # ===================================================
    # APPLICATION SUMMARY
    # ===================================================

    st.divider()

    st.subheader("📄 Application Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)


    with summary_col1:

        st.write(
            f"**Applicant Income:** ₹{applicant_income:,.0f}"
        )

        st.write(
            f"**Co-applicant Income:** ₹{coapplicant_income:,.0f}"
        )

        st.write(
            f"**Credit Score:** {credit_score}"
        )


    with summary_col2:

        st.write(
            f"**Loan Amount:** ₹{loan_amount:,.0f}"
        )

        st.write(
            f"**Loan Term:** {loan_term_years} years"
        )

        st.write(
            f"**Interest Rate:** {interest_rate:.1f}%"
        )


    with summary_col3:

        st.write(
            f"**Employment:** "
            f"{employment_status.replace('_', ' ').title()}"
        )

        st.write(
            f"**Property Area:** "
            f"{property_area.title()}"
        )

        st.write(
            f"**Dependents:** {dependents}"
        )


    # ===================================================
    # LOAN PAYMENT SUMMARY - ALWAYS SHOWN
    # ===================================================

    st.divider()

    st.subheader("💳 Estimated Loan Payment Summary")

    total_payment = emi * loan_term_months

    total_interest = total_payment - loan_amount


    pay_col1, pay_col2, pay_col3 = st.columns(3)


    with pay_col1:

        st.metric(
            "Loan Principal",
            f"₹{loan_amount:,.2f}"
        )


    with pay_col2:

        st.metric(
            "Total Interest",
            f"₹{total_interest:,.2f}"
        )


    with pay_col3:

        st.metric(
            "Total Amount Payable",
            f"₹{total_payment:,.2f}"
        )


    # ===================================================
    # FACTORS INFLUENCING DECISION
    # ===================================================

    if result.get("top_factors"):

        st.divider()

        st.subheader("💡 Factors Influencing the Decision")

        st.caption(
            "These features were identified as important "
            "by the machine learning model."
        )

        for factor in result["top_factors"]:

            feature = (
                factor["feature"]
                .replace("_", " ")
                .title()
            )

            if factor.get("impact", 0) > 0:

                st.success(
                    f"🟢 **{feature}** — Supports approval"
                )

            else:

                st.error(
                    f"🔴 **{feature}** — May reduce approval chances"
                )


    # ===================================================
    # SAVE PREDICTION HISTORY
    # ===================================================

    st.session_state.history.append({

        "decision": result["decision"],

        "probability": probability,

        "loan_amount": loan_amount,

        "loan_term_years": loan_term_years,

        "credit_score": credit_score,

        "interest_rate": interest_rate,

        "emi": emi
    })


# ===================================================
# PREDICTION HISTORY
# ===================================================

if st.session_state.history:

    st.divider()

    st.subheader("📜 Recent Predictions")


    for i, item in enumerate(
        reversed(st.session_state.history[-5:]),
        start=1
    ):

        with st.expander(
            f"Prediction {i} — "
            f"{item['decision']} "
            f"({item['probability']:.0%})"
        ):

            hist_col1, hist_col2, hist_col3, hist_col4 = (
                st.columns(4)
            )


            with hist_col1:

                st.metric(
                    "Decision",
                    item["decision"]
                )


            with hist_col2:

                st.metric(
                    "Probability",
                    f"{item['probability']:.1%}"
                )


            with hist_col3:

                st.metric(
                    "Loan Amount",
                    f"₹{item['loan_amount']:,.0f}"
                )


            with hist_col4:

                st.metric(
                    "Monthly EMI",
                    f"₹{item['emi']:,.2f}"
                )


# ===================================================
# FOOTER
# ===================================================

st.markdown(
    """
    <div class="footer">
        🏦 Loan Approval Prediction System<br>
        Machine Learning Project · Educational Purpose Only
    </div>
    """,
    unsafe_allow_html=True
)
