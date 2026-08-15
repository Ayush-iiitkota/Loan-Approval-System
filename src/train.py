"""Train, compare, and persist loan-approval models.

Usage:
    python -m src.train [--data data/raw/loan_data.csv] [--out models/model.joblib]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocess import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    prepare,
)

try:  # XGBoost is preferred but optional
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False


def build_candidates() -> dict[str, Pipeline]:
    numeric = NUMERIC_FEATURES + ENGINEERED_FEATURES
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    candidates: dict[str, object] = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42),
    }
    if HAS_XGB:
        candidates["xgboost"] = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
    else:
        candidates["gradient_boosting"] = GradientBoostingClassifier(random_state=42)
    return {name: Pipeline([("pre", pre), ("clf", clf)]) for name, clf in candidates.items()}


def evaluate(model: Pipeline, X, y) -> dict[str, float]:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
    }


def main(data_path: str, out_path: str) -> None:
    df = prepare(pd.read_csv(data_path))
    features = NUMERIC_FEATURES + ENGINEERED_FEATURES + CATEGORICAL_FEATURES
    X, y = df[features], df[TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    results: dict[str, dict[str, float]] = {}
    best_name, best_model, best_auc = None, None, -1.0
    for name, pipe in build_candidates().items():
        pipe.fit(X_tr, y_tr)
        metrics = evaluate(pipe, X_te, y_te)
        results[name] = metrics
        print(f"{name:>22}: {metrics}")
        if metrics["roc_auc"] > best_auc:
            best_name, best_model, best_auc = name, pipe, metrics["roc_auc"]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out)
    info = {
        "selected_model": best_name,
        "metrics": results,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "version": "1.0.0",
    }
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"\nSelected '{best_name}' (ROC-AUC {best_auc}). Saved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/raw/loan_data.csv")
    p.add_argument("--out", default="models/model.joblib")
    a = p.parse_args()
    main(a.data, a.out)
