"""FastAPI service for loan-approval predictions.

Run:
    uvicorn src.api:app --reload
Docs:
    http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .predict import MODEL_PATH, load_model, predict_application

app = FastAPI(
    title="try-self — Loan Approval API",
    description="Predict loan approval decisions with per-decision explanations.",
    version=__version__,
)


class Application(BaseModel):
    applicant_income: float = Field(..., ge=0, examples=[5400])
    coapplicant_income: float = Field(0, ge=0, examples=[1200])
    loan_amount: float = Field(..., gt=0, examples=[128000])
    loan_term_months: int = Field(..., gt=0, examples=[360])
    credit_history: int = Field(..., ge=0, le=1, examples=[1])
    dependents: int = Field(0, ge=0, examples=[0])
    employment_status: Literal["salaried", "self_employed", "unemployed"] = "salaried"
    property_area: Literal["urban", "semiurban", "rural"] = "urban"


class Factor(BaseModel):
    feature: str
    impact: float


class Prediction(BaseModel):
    decision: Literal["Approved", "Rejected"]
    probability: float
    threshold: float
    top_factors: list[Factor]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model/info")
def model_info() -> dict:
    meta = MODEL_PATH.with_suffix(".json")
    if not meta.exists():
        raise HTTPException(404, "Model not trained yet. Run `python -m src.train`.")
    return json.loads(meta.read_text())


@app.post("/predict", response_model=Prediction)
def predict(application: Application) -> dict:
    try:
        load_model()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    return predict_application(application.model_dump())
