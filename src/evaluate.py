"""Evaluation report: metrics, threshold sweep, and group fairness checks.

Usage:
    python -m src.evaluate [--data data/raw/loan_data.csv] [--model models/model.joblib]
"""
from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from .preprocess import TARGET, feature_columns, prepare


def threshold_sweep(y, proba, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "approval_rate": round(pred.mean(), 3),
                "precision": round(precision_score(y, pred, zero_division=0), 3),
                "recall": round(recall_score(y, pred, zero_division=0), 3),
            }
        )
    return pd.DataFrame(rows)


def group_report(df: pd.DataFrame, proba, group_col: str, threshold: float = 0.5) -> pd.DataFrame:
    out = df[[group_col, TARGET]].copy()
    out["predicted"] = (proba >= threshold).astype(int)
    g = out.groupby(group_col).agg(
        n=(TARGET, "size"),
        actual_approval_rate=(TARGET, "mean"),
        predicted_approval_rate=("predicted", "mean"),
    )
    return g.round(3)


def main(data_path: str, model_path: str) -> None:
    df = prepare(pd.read_csv(data_path))
    model = joblib.load(model_path)
    X, y = df[feature_columns()], df[TARGET].astype(int)
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    print("== Overall metrics ==")
    print(f"accuracy:  {accuracy_score(y, pred):.4f}")
    print(f"precision: {precision_score(y, pred):.4f}")
    print(f"recall:    {recall_score(y, pred):.4f}")
    print(f"roc_auc:   {roc_auc_score(y, proba):.4f}")

    print("\n== Threshold sweep ==")
    print(threshold_sweep(y, proba).to_string(index=False))

    for col in ("property_area", "employment_status"):
        print(f"\n== Group report: {col} ==")
        print(group_report(df, proba, col).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/raw/loan_data.csv")
    p.add_argument("--model", default="models/model.joblib")
    a = p.parse_args()
    main(a.data, a.model)
