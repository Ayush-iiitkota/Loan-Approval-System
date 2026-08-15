"""Generate a realistic synthetic loan-application dataset (data/raw/loan_data.csv)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 2000

employment = rng.choice(["salaried", "self_employed", "unemployed"], N, p=[0.62, 0.3, 0.08])
area = rng.choice(["urban", "semiurban", "rural"], N, p=[0.42, 0.35, 0.23])
income = np.round(rng.lognormal(8.45, 0.5, N), 0)          # ~ $3k-15k monthly
co_income = np.where(rng.random(N) < 0.55, np.round(rng.lognormal(7.6, 0.7, N), 0), 0)
loan_amount = np.round(rng.lognormal(11.6, 0.45, N), -3)   # ~ $40k-300k
term = rng.choice([120, 180, 240, 300, 360], N, p=[0.05, 0.1, 0.15, 0.1, 0.6])
credit = (rng.random(N) < 0.78).astype(int)
dependents = rng.choice([0, 1, 2, 3], N, p=[0.5, 0.25, 0.17, 0.08])

total = income + co_income
dti = (loan_amount / term) / np.maximum(total / 12, 1)
logit = (
    2.6 * credit
    - 2.1 * dti
    + 0.35 * np.log1p(total) - 3.0
    - 0.9 * (employment == "unemployed")
    + 0.25 * (area == "semiurban")
    - 0.12 * dependents
)
p = 1 / (1 + np.exp(-logit))
approved = (rng.random(N) < p).astype(int)

df = pd.DataFrame(
    {
        "applicant_income": income,
        "coapplicant_income": co_income,
        "loan_amount": loan_amount,
        "loan_term_months": term,
        "credit_history": credit,
        "dependents": dependents,
        "employment_status": employment,
        "property_area": area,
        "approved": approved,
    }
)
df.to_csv("data/raw/loan_data.csv", index=False)
print(f"Wrote {len(df)} rows, approval rate {approved.mean():.2%}")
