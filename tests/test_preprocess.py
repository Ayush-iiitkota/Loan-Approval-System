import pandas as pd
import pytest

from src.preprocess import clean, engineer_features, prepare


@pytest.fixture
def raw_df():
    return pd.DataFrame(
        [
            {  # valid row
                "applicant_income": 5000, "coapplicant_income": 1000,
                "loan_amount": 120000, "loan_term_months": 360,
                "credit_history": 1, "dependents": 0,
                "employment_status": " Salaried ", "property_area": "URBAN",
                "approved": 1,
            },
            {  # invalid: negative income
                "applicant_income": -10, "coapplicant_income": 0,
                "loan_amount": 50000, "loan_term_months": 360,
                "credit_history": 1, "dependents": 0,
                "employment_status": "salaried", "property_area": "rural",
                "approved": 0,
            },
            {  # invalid: unknown category
                "applicant_income": 4000, "coapplicant_income": 0,
                "loan_amount": 50000, "loan_term_months": 360,
                "credit_history": 1, "dependents": 0,
                "employment_status": "astronaut", "property_area": "rural",
                "approved": 0,
            },
        ]
    )


def test_clean_drops_invalid_rows(raw_df):
    out = clean(raw_df)
    assert len(out) == 1
    assert out.loc[0, "employment_status"] == "salaried"
    assert out.loc[0, "property_area"] == "urban"


def test_engineer_features_adds_columns(raw_df):
    out = engineer_features(clean(raw_df))
    for col in ("total_income", "dti_ratio", "loan_to_income"):
        assert col in out.columns
    assert out.loc[0, "total_income"] == 6000
    assert out.loc[0, "dti_ratio"] > 0


def test_prepare_end_to_end(raw_df):
    out = prepare(raw_df)
    assert len(out) == 1 and "dti_ratio" in out.columns
