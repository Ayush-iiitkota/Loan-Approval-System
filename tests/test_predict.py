import pytest

from src.predict import load_model, predict_application

VALID = {
    "applicant_income": 5400, "coapplicant_income": 1200,
    "loan_amount": 128000, "loan_term_months": 360,
    "credit_history": 1, "dependents": 0,
    "employment_status": "salaried", "property_area": "urban",
}


@pytest.fixture(autouse=True)
def require_model():
    try:
        load_model()
    except FileNotFoundError:
        pytest.skip("Model not trained; run `python -m src.train` first.")


def test_predict_returns_expected_shape():
    result = predict_application(VALID)
    assert result["decision"] in {"Approved", "Rejected"}
    assert 0.0 <= result["probability"] <= 1.0
    assert len(result["top_factors"]) > 0
    assert {"feature", "impact"} <= set(result["top_factors"][0])


def test_bad_credit_lowers_probability():
    good = predict_application(VALID)["probability"]
    bad = predict_application({**VALID, "credit_history": 0})["probability"]
    assert bad <= good


def test_missing_field_raises():
    incomplete = {k: v for k, v in VALID.items() if k != "loan_amount"}
    with pytest.raises(ValueError):
        predict_application(incomplete)
