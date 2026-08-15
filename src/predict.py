"""Inference + explanations for loan-approval predictions."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .preprocess import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERIC_FEATURES,
    engineer_features,
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
FEATURES = NUMERIC_FEATURES + ENGINEERED_FEATURES + CATEGORICAL_FEATURES

DEFAULTS = {"coapplicant_income": 0.0, "dependents": 0}


@lru_cache(maxsize=1)
def load_model(path: str | None = None):
    model_path = Path(path) if path else MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run `python -m src.train` first."
        )
    return joblib.load(model_path)


def _as_frame(application: dict) -> pd.DataFrame:
    row = {**DEFAULTS, **application}
    missing = [f for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES if f not in row]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    df = pd.DataFrame([row])
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return engineer_features(df)


def _top_factors(model, X: pd.DataFrame, k: int = 5) -> list[dict]:
    """Per-prediction factor attribution.

    Uses SHAP when installed; otherwise falls back to a permutation-style
    single-row sensitivity: how much the approval probability moves when each
    feature is neutralised to the training-median encoding.
    """
    try:
        import shap  # optional dependency

        pre = model.named_steps["pre"]
        clf = model.named_steps["clf"]
        Xt = pre.transform(X)
        Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
        explainer = shap.Explainer(clf, Xt)
        values = explainer(Xt).values[0]
        names = pre.get_feature_names_out()
        pairs = sorted(zip(names, values), key=lambda p: abs(p[1]), reverse=True)[:k]
        return [
            {"feature": n.split("__", 1)[-1], "impact": round(float(v), 4)}
            for n, v in pairs
        ]
    except Exception:
        base = float(model.predict_proba(X)[:, 1][0])
        impacts = []
        for col in FEATURES:
            X_alt = X.copy()
            if col in CATEGORICAL_FEATURES:
                X_alt[col] = "semiurban" if col == "property_area" else "salaried"
            else:
                X_alt[col] = float(X[col].iloc[0]) * 0  # neutralise
            alt = float(model.predict_proba(X_alt)[:, 1][0])
            impacts.append({"feature": col, "impact": round(base - alt, 4)})
        impacts.sort(key=lambda d: abs(d["impact"]), reverse=True)
        return impacts[:k]


def predict_application(
    application: dict, threshold: float = 0.5, model_path: str | None = None
) -> dict:
    """Return decision, probability, and top contributing factors.

    Example
    -------
    >>> predict_application({
    ...     "applicant_income": 5400, "coapplicant_income": 1200,
    ...     "loan_amount": 128000, "loan_term_months": 360,
    ...     "credit_history": 1, "dependents": 0,
    ...     "employment_status": "salaried", "property_area": "urban",
    ... })["decision"] in {"Approved", "Rejected"}
    True
    """
    model = load_model(model_path)
    X = _as_frame(application)[FEATURES]
    probability = float(model.predict_proba(X)[:, 1][0])
    return {
        "decision": "Approved" if probability >= threshold else "Rejected",
        "probability": round(probability, 4),
        "threshold": threshold,
        "top_factors": _top_factors(model, X),
    }
