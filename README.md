# Telco Customer Churn Prediction

[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)

A machine learning project that predicts whether a telecommunications customer will churn (cancel service) using demographic, account, and service information. Includes EDA, feature engineering, trained models (Logistic Regression, Random Forest), and a Streamlit inference app.

## Quick Start

```bash
# Clone and enter the project
cd telcom_churn_project

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app/streamlit_app.py
```

## Project Structure

```
telcom_churn_project/
├── app/
│   ├── streamlit_app.py          # Streamlit inference UI
│   └── requirements.txt          # App-specific dependencies
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Original dataset
│   └── processed/
│       └── telco_dataset.csv     # Cleaned, engineered dataset
├── models/                       # Trained artifacts (also duplicated in app/models/)
│   ├── logistic_regression_model.joblib
│   ├── random_forest_model.joblib
│   ├── scaler.joblib
│   └── selected_features.pkl
├── notebooks/
│   ├── eda_churn.ipynb           # Exploratory data analysis
│   ├── feature_engineering.ipynb # Data cleaning & feature creation
│   └── model_comparison.ipynb    # Training, tuning, evaluation
├── requirements.txt              # Project dependencies
└── README.md
```

> **Note:** The `models/` directory is duplicated inside `app/models/` for Streamlit Cloud deployment. Both copies are identical.

## Data

- **Source:** [WA_Fn-UseC_-Telco-Customer-Churn.csv](data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv) — 7,043 customers, 21 columns, ~26.5% churn rate.
- **Processed:** [data/processed/telco_dataset.csv](data/processed/telco_dataset.csv) — output of `notebooks/feature_engineering.ipynb`.

To re-run the pipeline from scratch, start with the raw CSV and follow `notebooks/feature_engineering.ipynb`.

## Model Evaluation

Both models were trained on 33 features, then re-trained on 17 features selected via `SelectFromModel` (median threshold). Hyperparameter tuning used 3-fold cross-validation with ROC AUC scoring.

| Model | Features | Accuracy | Precision (Churn) | Recall (Churn) | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 33 (all) | 0.793 | 0.641 | 0.505 | **0.836** |
| Random Forest | 33 (all) | 0.797 | 0.662 | 0.487 | **0.829** |
| Logistic Regression | **17 (selected)** | 0.794 | 0.641 | 0.511 | **0.832** |
| Random Forest | **17 (selected)** | 0.797 | 0.662 | 0.487 | **0.829** |

**Selected features** (17): `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `MonthlyToTotalRatio`, `TenureGroup`, `InternetService_Fiber optic`, `InternetService_DSL`, `Contract_One year`, `Contract_Two year`, `PaymentMethod_Electronic check`, `PaymentMethod_Mailed check`, `OnlineSecurity_Yes`.

Feature importance (Random Forest) is dominated by `MonthlyToTotalRatio`, `tenure`, `TotalCharges`, and `MonthlyCharges`.

## Streamlit App (`app/streamlit_app.py`)

The app loads a **Logistic Regression** model and a pre-fitted `StandardScaler` to serve live predictions through a browser UI.

### Features

- **Tabbed input form** — Demographics, Account Info, Services
- **Churn risk gauge** — Plotly gauge chart with LOW (≤0.4), MEDIUM (≤0.7), HIGH (>0.7) thresholds
- **Customer summary** — Side-by-side input/output overview
- **Retention recommendations** — Contextual suggestions (contract offers, loyalty bonuses, bundle discounts)
- **Industry comparison** — Benchmarks prediction against average churn (~25%)
- **Downloadable report** — Export prediction results as CSV

### Limitations

| Limitation | Details |
|---|---|
| **Only Logistic Regression** | The model selector in the app is hardcoded to Logistic Regression; Random Forest is not wired for inference. |
| **TenureGroup mismatch** | Training uses 6 bins (`0-1yr` through `5-6yr`), but the app only defines 5. Customers with tenure ≥48 are mapped to the wrong bin. |
| **No retraining UI** | The app is inference-only; there is no feedback loop to retrain on new data. |

## Inference Example (Python)

A self-contained example using the 17 selected features:

```python
import joblib
import pandas as pd
import numpy as np

# Load artifacts
scaler = joblib.load("models/scaler.joblib")
model = joblib.load("models/logistic_regression_model.joblib")

# Single customer input (all 17 selected features must be present)
row = pd.DataFrame([{
    "tenure": 12,
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "PhoneService": 1,
    "PaperlessBilling": 1,
    "MonthlyToTotalRatio": 70.0 / 840.0,
    "TenureGroup": 1,                        # 0-1yr -> 0, 1-2yr -> 1, ..., 5-6yr -> 5
    "InternetService_Fiber optic": 1,
    "InternetService_DSL": 0,
    "Contract_One year": 0,
    "Contract_Two year": 0,
    "PaymentMethod_Electronic check": 1,
    "PaymentMethod_Mailed check": 0,
    "OnlineSecurity_Yes": 0,
}])

# Scale numeric features
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "MonthlyToTotalRatio"]
row[numeric_cols] = scaler.transform(row[numeric_cols])

churn_prob = model.predict_proba(row)[0, 1]
print(f"Churn probability: {churn_prob:.3f}")
print(f"Prediction: {'Churn' if churn_prob >= 0.5 else 'No Churn'}")
```

## Reproducing Training

1. Explore the data: `notebooks/eda_churn.ipynb`
2. Clean and engineer features: `notebooks/feature_engineering.ipynb`
   - Run from `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
   - Outputs `data/processed/telco_dataset.csv`
3. Train and evaluate: `notebooks/model_comparison.ipynb`
   - Trains both models, performs feature selection, saves artifacts to `models/`

## Known Issues

1. **`ServiceCount` is always 0** — In `feature_engineering.ipynb`, the feature is computed before string columns are encoded to integers, so the comparison `== 1` always evaluates to `False`. This feature was dropped during feature selection, so it does not affect the final model.
2. **TenureGroup bin mismatch** — The app uses 5 bins while training uses 6. Should be aligned to avoid silent prediction degradation.
3. **Duplicate model artifacts** — `models/` and `app/models/` contain identical copies. A symlink or CI step could eliminate duplication.
4. **No `.gitignore`** — Binary artifacts (`.joblib`, `.pkl`) and `.ipynb_checkpoints` directories are tracked. Consider adding a `.gitignore`.

## Roadmap

- [ ] Wire Random Forest into the Streamlit app
- [ ] Fix `TenureGroup` mismatch between app and training
- [ ] Add Dockerfile / docker-compose for containerized deployment
- [ ] Set up CI (linting, test notebooks, model validation)
- [ ] Add logging and monitoring for production predictions
- [ ] Pin dependency versions in `requirements.txt`

## Contributing

Contributions are welcome. For major changes (new features, refactors, or fixes), please open an issue first to discuss.

## License

This project is provided for educational and portfolio purposes. No license is specified.
