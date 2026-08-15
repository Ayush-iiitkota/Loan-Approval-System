"""Data cleaning and feature engineering for loan-approval prediction."""
from __future__ import annotations

import pandas as pd

CATEGORICAL_FEATURES = ["employment_status", "property_area"]
NUMERIC_FEATURES = [
    "applicant_income",
    "coapplicant_income",
    "loan_amount",
    "loan_term_months",
    "credit_history",
    "dependents",
]
ENGINEERED_FEATURES = ["total_income", "dti_ratio", "loan_to_income"]
TARGET = "approved"

VALID_EMPLOYMENT = {"salaried", "self_employed", "unemployed"}
VALID_AREAS = {"urban", "semiurban", "rural"}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Safe to call on a single-row frame at inference."""
    df = df.copy()
    df["total_income"] = df["applicant_income"] + df["coapplicant_income"]
    monthly_payment = df["loan_amount"] / df["loan_term_months"].clip(lower=1)
    monthly_income = (df["total_income"] / 12).clip(lower=1)
    df["dti_ratio"] = (monthly_payment / monthly_income).clip(upper=10)
    df["loan_to_income"] = (df["loan_amount"] / df["total_income"].clip(lower=1)).clip(upper=50)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: types, valid categories, sensible ranges, drop bad rows."""
    df = df.copy()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=NUMERIC_FEATURES)
    df = df[df["applicant_income"] >= 0]
    df = df[df["loan_amount"] > 0]
    df = df[df["loan_term_months"] > 0]
    df["credit_history"] = df["credit_history"].clip(0, 1).round().astype(int)
    df["employment_status"] = df["employment_status"].astype(str).str.lower().str.strip()
    df["property_area"] = df["property_area"].astype(str).str.lower().str.strip()
    df = df[df["employment_status"].isin(VALID_EMPLOYMENT)]
    df = df[df["property_area"].isin(VALID_AREAS)]
    return df.reset_index(drop=True)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline: clean + engineer."""
    return engineer_features(clean(df))


def feature_columns() -> list[str]:
    return NUMERIC_FEATURES + ENGINEERED_FEATURES + CATEGORICAL_FEATURES
