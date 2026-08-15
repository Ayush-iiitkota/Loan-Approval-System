import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.predict import load_model

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_predict_endpoint():
    try:
        load_model()
    except FileNotFoundError:
        pytest.skip("Model not trained")
    r = client.post(
        "/predict",
        json={
            "applicant_income": 5400, "coapplicant_income": 1200,
            "loan_amount": 128000, "loan_term_months": 360,
            "credit_history": 1, "dependents": 0,
            "employment_status": "salaried", "property_area": "urban",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in {"Approved", "Rejected"}


def test_predict_validation_error():
    r = client.post("/predict", json={"applicant_income": -5})
    assert r.status_code == 422
