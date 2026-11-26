
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re

from sklearn.base import BaseEstimator, TransformerMixin


st.set_page_config(
    page_title="FinBridge | Credit Intelligence",
    page_icon="💳",
    layout="wide"
)


# ---------- FEATURE ENGINEERING ----------

class FinBridgeFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(
            columns=["application_id", "applicant_ref_id"],
            errors="ignore"
        )

        original_columns = list(X.columns)

        for col in original_columns:
            X[f"{col}_missing"] = X[col].isna().astype("int8")

        for col in [
            "interest_rate_pct",
            "revolving_utilization_pct"
        ]:
            if col in X.columns:
                X[col] = (
                    X[col].astype(str)
                    .str.replace("%", "", regex=False)
                    .replace({
                        "nan": np.nan,
                        "None": np.nan,
                        "": np.nan
                    })
                )
                X[col] = pd.to_numeric(X[col], errors="coerce")

        numeric_candidates = [
            "requested_amount", "monthly_emi", "annual_income",
            "debt_to_income_ratio", "delinquencies_24m",
            "credit_inquiries_6m", "active_credit_lines",
            "derogatory_public_records", "revolving_credit_balance",
            "total_credit_accounts", "recent_payment_amount",
            "interest_rate_pct", "revolving_utilization_pct"
        ]

        for col in numeric_candidates:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        if "risk_subtier" in X.columns:

            def subtier_number(value):
                if pd.isna(value):
                    return np.nan

                match = re.fullmatch(
                    r"\s*([A-G])\s*([1-5])\s*",
                    str(value).upper()
                )

                if not match:
                    return np.nan

                letter = ord(match.group(1)) - ord("A")
                digit = int(match.group(2))

                return letter * 5 + digit

            X["risk_subtier_number"] = (
                X["risk_subtier"].map(subtier_number)
            )

            X = X.drop(columns=["risk_subtier"])

        income = (
            X["annual_income"].replace(0, np.nan)
            if "annual_income" in X.columns
            else np.nan
        )

        if "monthly_emi" in X.columns and "annual_income" in X.columns:
            X["yearly_emi_to_annual_income"] = (
                X["monthly_emi"] * 12
            ) / income

        if "requested_amount" in X.columns and "annual_income" in X.columns:
            X["requested_amount_to_annual_income"] = (
                X["requested_amount"] / income
            )

        if (
            "revolving_credit_balance" in X.columns
            and "annual_income" in X.columns
        ):
            X["revolving_balance_to_annual_income"] = (
                X["revolving_credit_balance"] / income
            )

        if (
            "revolving_credit_balance" in X.columns
            and "total_credit_accounts" in X.columns
        ):
            accounts = X["total_credit_accounts"].replace(0, np.nan)

            X["revolving_balance_per_credit_account"] = (
                X["revolving_credit_balance"] / accounts
            )

        if (
            "delinquencies_24m" in X.columns
            and "credit_inquiries_6m" in X.columns
        ):
            X["delinquency_inquiry_total"] = (
                X["delinquencies_24m"].fillna(0)
                + X["credit_inquiries_6m"].fillna(0)
            )

        X = X.replace([np.inf, -np.inf], np.nan)

        return X


# ---------- DESIGN AND WHITE TEXT THEME ----------

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0b1220;
    color: #ffffff !important;
}

/* All general text */
.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp li,
.stApp div {
    color: #ffffff;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111c2e;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

/* Captions */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {
    color: #ffffff !important;
}

/* Metric cards */
.metric-card {
    background: #15243a;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #263b55;
}

.metric-label {
    color: #ffffff !important;
    font-size: 14px;
}

.metric-value {
    color: #4dd6b0 !important;
    font-size: 28px;
    font-weight: bold;
}

.small-note {
    color: #ffffff !important;
    font-size: 13px;
}

/* Input boxes */
.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stTimeInput input,
[data-baseweb="select"] > div {
    color: #ffffff !important;
    background-color: #15243a !important;
}

/* Dropdown text */
[data-baseweb="popover"],
[data-baseweb="menu"],
[data-baseweb="menu"] *,
[role="listbox"],
[role="option"] {
    color: #ffffff !important;
    background-color: #15243a !important;
}

/* Selectbox labels */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] * {
    color: #ffffff !important;
}

/* Buttons */
div.stButton > button,
div.stFormSubmitButton > button {
    background-color: #16a889;
    color: #ffffff !important;
    border-radius: 8px;
    border: none;
    height: 45px;
    font-weight: bold;
}

/* Buttons on hover */
div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background-color: #0e8f75;
    color: #ffffff !important;
    border: 1px solid #4dd6b0;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    color: #ffffff !important;
}

/* Expanders */
[data-testid="stExpander"] {
    border: 1px solid #263b55;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: #ffffff !important;
}

/* Tabs */
button[data-baseweb="tab"],
button[data-baseweb="tab"] * {
    color: #ffffff !important;
}

/* Info and warning text */
[data-testid="stAlert"] * {
    color: #ffffff !important;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background-color: #16a889;
}

/* Radio buttons */
[data-testid="stRadio"] label,
[data-testid="stRadio"] p {
    color: #ffffff !important;
}

/* Teal brand heading */
.brand-heading {
    color: #4dd6b0 !important;
    letter-spacing: 3px;
}

</style>
""", unsafe_allow_html=True)


# ---------- LOAD FILES ----------

@st.cache_resource
def load_models():
    return joblib.load("models/finbridge_models.pkl")


@st.cache_data
def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    return train, test


if not os.path.exists("models/finbridge_models.pkl"):
    st.error("Model file not found. Run python train_model.py first.")
    st.stop()

if not os.path.exists("train.csv") or not os.path.exists("test.csv"):
    st.error("Please place train.csv and test.csv in the project folder.")
    st.stop()

artifacts = load_models()
train, test = load_data()

selected_kind = artifacts["selected_kind"]
selected_name = artifacts["selected_name"]
models = artifacts


# ---------- SIDEBAR ----------

st.sidebar.title("FINBRIDGE")
st.sidebar.caption("CREDIT INTELLIGENCE PLATFORM")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Risk Prediction",
        "Model Performance",
        "Feature Importance",
        "Dataset Explorer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("FinBridge Capital")
st.sidebar.caption("Automated Credit Risk Assessment")


# ---------- HEADER ----------

st.markdown(
    "<p class='brand-heading'>FINBRIDGE CAPITAL</p>",
    unsafe_allow_html=True
)


# ---------- OVERVIEW ----------

if page == "Overview":

    st.title("Credit Risk Dashboard")

    st.write(
        "Monitor loan applications and understand borrower default risk."
    )

    total = len(train)
    defaults = int(train["default_flag"].sum())
    repaid = total - defaults
    default_rate = defaults / total * 100

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        (c1, "Training Applications", f"{total:,}"),
        (c2, "Default Cases", f"{defaults:,}"),
        (c3, "Successfully Repaid", f"{repaid:,}"),
        (c4, "Default Rate", f"{default_rate:.2f}%")
    ]

    for col, label, value in cards:
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### Application Distribution")

    left, right = st.columns(2)

    with left:
        st.subheader("Loan Outcome Distribution")

        outcome = pd.Series({
            "Repaid": repaid,
            "Defaulted": defaults
        })

        st.bar_chart(outcome)

    with right:
        st.subheader("Default Distribution by Risk Tier")

        tier_data = (
            train.groupby("risk_tier")["default_flag"].mean() * 100
        )

        st.bar_chart(tier_data)

    st.markdown("---")
    st.subheader("Current Machine Learning Model")

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Selected model</div>
            <div class="metric-value">{selected_name}</div>
            <p class="small-note">
                Selected after comparing Logistic Regression,
                HistGradientBoosting and their rank blend.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------- RISK PREDICTION ----------

elif page == "Risk Prediction":

    st.title("Applicant Risk Assessment")

    st.write(
        "Enter borrower information to estimate their relative "
        "default risk score."
    )

    with st.form("prediction_form"):

        st.subheader("Personal and Financial Information")

        col1, col2 = st.columns(2)

        with col1:

            annual_income = st.number_input(
                "Annual Income ($)",
                min_value=1.0,
                value=65000.0,
                step=1000.0
            )

            requested_amount = st.number_input(
                "Requested Loan Amount ($)",
                min_value=1.0,
                value=12000.0,
                step=500.0
            )

            monthly_emi = st.number_input(
                "Monthly EMI ($)",
                min_value=0.0,
                value=350.0,
                step=10.0
            )

            interest_rate = st.number_input(
                "Interest Rate (%)",
                min_value=0.0,
                value=12.0,
                step=0.1
            )

            debt_ratio = st.number_input(
                "Debt-to-Income Ratio",
                min_value=0.0,
                value=15.0,
                step=0.5
            )

            repayment_tenure = st.selectbox(
                "Repayment Tenure",
                ["36 months", "60 months"]
            )

        with col2:

            risk_tier = st.selectbox(
                "Risk Tier",
                list("ABCDEFG"),
                index=1
            )

            risk_subtier = st.selectbox(
                "Risk Subtier",
                [
                    f"{letter}{number}"
                    for letter in "ABCDEFG"
                    for number in range(1, 6)
                ],
                index=7
            )

            employment = st.selectbox(
                "Employment Tenure",
                [
                    "< 1 year", "1 year", "2 years",
                    "3 years", "4 years", "5 years",
                    "6 years", "7 years", "8 years",
                    "9 years", "10+ years"
                ]
            )

            housing = st.selectbox(
                "Housing Status",
                ["RENT", "OWN", "MORTGAGE", "OTHER"]
            )

            verification = st.selectbox(
                "Income Verification",
                [
                    "Verified",
                    "Not Verified",
                    "Source Verified"
                ]
            )

            purpose = st.selectbox(
                "Loan Purpose",
                [
                    "debt_consolidation",
                    "credit_card",
                    "home_improvement",
                    "major_purchase",
                    "medical",
                    "car",
                    "other"
                ]
            )

        st.subheader("Credit History")

        col3, col4 = st.columns(2)

        with col3:

            delinquencies = st.number_input(
                "Delinquencies in 24 Months",
                min_value=0,
                value=0
            )

            inquiries = st.number_input(
                "Credit Inquiries in 6 Months",
                min_value=0,
                value=1
            )

            active_lines = st.number_input(
                "Active Credit Lines",
                min_value=0,
                value=7
            )

            public_records = st.number_input(
                "Derogatory Public Records",
                min_value=0,
                value=0
            )

        with col4:

            revolving_balance = st.number_input(
                "Revolving Credit Balance ($)",
                min_value=0.0,
                value=5000.0
            )

            utilization = st.number_input(
                "Revolving Utilization (%)",
                min_value=0.0,
                max_value=100.0,
                value=40.0
            )

            total_accounts = st.number_input(
                "Total Credit Accounts",
                min_value=1,
                value=15
            )

            recent_payment = st.number_input(
                "Recent Payment Amount ($)",
                min_value=0.0,
                value=300.0
            )

        submitted = st.form_submit_button(
            "ANALYZE CREDIT RISK",
            use_container_width=True
        )

    if submitted:

        applicant = pd.DataFrame([{
            "application_id": "NEW_APPLICATION",
            "applicant_ref_id": 0,
            "requested_amount": requested_amount,
            "repayment_tenure": repayment_tenure,
            "interest_rate_pct": f"{interest_rate}%",
            "monthly_emi": monthly_emi,
            "risk_tier": risk_tier,
            "risk_subtier": risk_subtier,
            "employment_tenure": employment,
            "housing_status": housing,
            "annual_income": annual_income,
            "income_verification": verification,
            "loan_purpose": purpose,
            "debt_to_income_ratio": debt_ratio,
            "delinquencies_24m": delinquencies,
            "credit_inquiries_6m": inquiries,
            "active_credit_lines": active_lines,
            "derogatory_public_records": public_records,
            "revolving_credit_balance": revolving_balance,
            "revolving_utilization_pct": f"{utilization}%",
            "total_credit_accounts": total_accounts,
            "recent_payment_amount": recent_payment
        }])

        with st.spinner("Analyzing applicant..."):

            if selected_kind == "hgb":

                raw_score = (
                    models["hgb_model"]
                    .predict_proba(applicant)[:, 1][0]
                )

                reference = models["oof_scores"]["hgb"]

            elif selected_kind == "logistic":

                raw_score = (
                    models["logistic_model"]
                    .predict_proba(applicant)[:, 1][0]
                )

                reference = models["oof_scores"]["logistic"]

            else:

                log_score = (
                    models["logistic_model"]
                    .predict_proba(applicant)[:, 1][0]
                )

                hgb_score = (
                    models["hgb_model"]
                    .predict_proba(applicant)[:, 1][0]
                )

                log_rank = np.searchsorted(
                    np.sort(models["oof_scores"]["logistic"]),
                    log_score,
                    side="right"
                ) / len(models["oof_scores"]["logistic"])

                hgb_rank = np.searchsorted(
                    np.sort(models["oof_scores"]["hgb"]),
                    hgb_score,
                    side="right"
                ) / len(models["oof_scores"]["hgb"])

                raw_score = (log_rank + hgb_rank) / 2
                reference = None

            if selected_kind != "blend":

                score = np.searchsorted(
                    np.sort(reference),
                    raw_score,
                    side="right"
                ) / len(reference)

            else:
                score = raw_score

        score = float(np.clip(score, 0.000001, 0.999999))

        st.markdown("---")
        st.subheader("Risk Assessment Result")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Relative Default Risk Score",
                f"{score * 100:.2f}/100"
            )

        with c2:
            st.metric("Model Used", selected_name)

        st.progress(score)

        if score >= 0.80:
            st.error("High relative default risk")

        elif score >= 0.50:
            st.warning("Moderate relative default risk")

        else:
            st.success("Low relative default risk")

        st.caption(
            "This is a rank-scaled risk score, not a calibrated "
            "probability of default. It is intended as a decision "
            "support tool, not an automatic lending decision."
        )


# ---------- MODEL PERFORMANCE ----------

elif page == "Model Performance":

    st.title("Model Performance")

    st.write(
        "Validation results from the trained machine learning models."
    )

    results_path = "outputs/validation_metrics.csv"

    if os.path.exists(results_path):

        results = pd.read_csv(results_path)

        st.dataframe(
            results.style.format({
                "ROC-AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "threshold": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("ROC-AUC Comparison")

        st.bar_chart(
            results.set_index("model")[["ROC-AUC"]]
        )

        st.subheader("Precision, Recall and F1")

        st.bar_chart(
            results.set_index("model")[
                ["Precision", "Recall", "F1"]
            ]
        )

        st.info(
            f"Selected model: {selected_name}. "
            "The selection is based on five-fold out-of-fold ROC-AUC."
        )

    else:
        st.warning("Validation metrics not found.")


# ---------- FEATURE IMPORTANCE ----------

elif page == "Feature Importance":

    st.title("Feature Importance")

    st.write(
        "Understand which applicant attributes most influence "
        "the model's predictive performance."
    )

    path = "outputs/feature_importance.csv"

    if os.path.exists(path):

        importance = pd.read_csv(path)

        importance = importance.sort_values(
            "importance",
            ascending=False
        ).head(10)

        st.subheader("Top 10 Important Features")

        st.bar_chart(
            importance.set_index("feature")[["importance"]]
        )

        explanations = {
            "recent_payment_amount":
                "Recent payment activity may provide information "
                "about a borrower's repayment behaviour.",

            "monthly_emi":
                "The monthly installment indicates the borrower's "
                "scheduled repayment burden.",

            "repayment_tenure":
                "The loan term determines how long the borrower "
                "must make repayments.",

            "interest_rate_pct":
                "The interest rate affects the total cost of borrowing.",

            "annual_income":
                "Income indicates the borrower's financial capacity.",

            "requested_amount":
                "The requested principal represents the amount "
                "of credit being sought.",

            "loan_purpose":
                "The intended use of the loan may reflect different "
                "borrowing needs.",

            "revolving_utilization_pct":
                "Credit utilization measures how much revolving "
                "credit is currently being used.",

            "risk_subtier":
                "The internal subgrade represents the lender's "
                "more detailed risk classification.",

            "credit_inquiries_6m":
                "Recent credit inquiries can indicate new credit "
                "seeking activity."
        }

        st.subheader("Business Interpretation")

        for _, row in importance.iterrows():

            feature = row["feature"]

            explanation = explanations.get(
                feature,
                "This attribute contributes to the model's predictions."
            )

            with st.expander(feature):

                st.write(explanation)

                st.write(
                    f"Permutation importance: {row['importance']:.5f}"
                )

        if "recent_payment_amount" in importance["feature"].values:
            st.warning(
                "Model audit: verify that recent_payment_amount "
                "is genuinely available before loan origination. "
                "If it reflects a payment made after origination, "
                "it could cause data leakage."
            )

    else:
        st.warning("Feature importance file not found.")


# ---------- DATASET EXPLORER ----------

elif page == "Dataset Explorer":

    st.title("Dataset Explorer")

    st.write(
        "Explore the anonymized loan application dataset."
    )

    st.subheader("Training Dataset")

    st.dataframe(
        train.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.write(f"Training records: {len(train):,}")
    st.write(f"Testing records: {len(test):,}")

    st.subheader("Default Rate by Risk Tier")

    tier_stats = train.groupby("risk_tier")[
        "default_flag"
    ].agg(["count", "mean"])

    tier_stats["mean"] *= 100

    tier_stats.columns = [
        "Applications",
        "Default Rate (%)"
    ]

    st.dataframe(
        tier_stats.style.format({
            "Default Rate (%)": "{:.2f}%"
        }),
        use_container_width=True
    )

    st.subheader("Missing Values")

    missing = train.isnull().sum()

    missing = missing[
        missing > 0
    ].sort_values(ascending=False)

    if len(missing):
        st.bar_chart(missing)

    else:
        st.success(
            "No missing values in the training dataset."
        )


# ---------- FOOTER ----------

st.markdown("---")

st.markdown(
    """
    <center style='color:#ffffff;'>
        FINBRIDGE CAPITAL | CREDIT RISK INTELLIGENCE
        <br>
        Machine Learning Powered Decision Support
    </center>
    """,
    unsafe_allow_html=True
)
