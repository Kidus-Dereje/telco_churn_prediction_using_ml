# Telco Customer Churn Prediction

This repository contains a machine learning project that analyzes customer churn for a telecommunications company and provides models and a simple Streamlit app for inference.

## Summary

The goal is to predict whether a customer will churn (cancel service) using customer demographic, account, and service information. This project contains exploratory data analysis, feature engineering and multiple trained models (logistic regression and random forest), plus a Streamlit app for quick inference.

## Project Structure

- `app/` — Streamlit application for model inference
	- `streamlit_app.py` — app entrypoint
- `data/`
	- `raw/` — original dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
	- `processed/` — cleaned / processed dataset used for modeling (`telco_dataset.csv`)
- `models/` — serialized artifacts
	- `logistic_regression_model.joblib`
	- `random_forest_model.joblib`
	- `scaler.joblib`
- `notebooks/` — EDA, feature engineering and model comparison notebooks
	- `eda_churn.ipynb`
	- `feature_engineering.ipynb`
	- `model_comparison.ipynb`
- `README.md` — this file

## Data

- Raw source: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (original Kaggle-like Telco churn dataset)
- Processed dataset used for modeling is in `data/processed/telco_dataset.csv`.

If you plan to re-run experiments, start from the raw CSV and follow the transformations in `notebooks/feature_engineering.ipynb`.

## Models

Two models and a scaler are included in `models/`:
- `logistic_regression_model.joblib` — a baseline logistic regression
- `random_forest_model.joblib` — a random forest classifier (typically stronger performance)
- `scaler.joblib` — feature scaler used during training (must be applied to numeric features before prediction)

See `notebooks/model_comparison.ipynb` for training details, hyperparameters and evaluation metrics (ROC AUC, accuracy, precision/recall, confusion matrix).

## Quickstart — Run the Streamlit app (recommended)

1. Create and activate a Python environment (example using venv):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate
```

2. Install dependencies (create `requirements.txt` if not present). Typical packages:

```bash
pip install streamlit pandas scikit-learn joblib numpy
```

3. Start the app:

```bash
streamlit run app/streamlit_app.py
```

The app will load artifacts from `models/` and provide a UI for making single-customer predictions and viewing model outputs.

## Inference Example (Python)

Load the scaler and model, preprocess inputs similarly to training, then predict:

```python
import joblib
import pandas as pd

# Load artifacts
scaler = joblib.load('models/scaler.joblib')
model = joblib.load('models/random_forest_model.joblib')

# Example input as DataFrame (single row)
row = pd.DataFrame([{ 
		'gender':'Female', 'SeniorCitizen':0, 'Partner':'Yes', 'Dependents':'No',
		'tenure':12, 'MonthlyCharges':70.0, 'TotalCharges':840.0,
		# include other features exactly as used during training
}])

# Apply same preprocessing used in training (dummy encoding, scaling numeric cols, etc.)
# numeric_cols = [...]
# row[numeric_cols] = scaler.transform(row[numeric_cols])

pred_proba = model.predict_proba(row)[:, 1]
prediction = (pred_proba >= 0.5).astype(int)

print('churn_probability:', float(pred_proba))
print('predicted_churn:', int(prediction))
```

Adapt the preprocessing steps to match what's in `notebooks/feature_engineering.ipynb`.

## Reproducing Training

1. Inspect `notebooks/eda_churn.ipynb` for data exploration and insights.
2. Follow `notebooks/feature_engineering.ipynb` to clean and transform `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` into `data/processed/telco_dataset.csv`.
3. Train and compare models in `notebooks/model_comparison.ipynb` and save artifacts with `joblib` to `models/`.

## Evaluation

Model evaluation (confusion matrix, precision/recall, ROC AUC) and comparison details are recorded in `notebooks/model_comparison.ipynb`. Use those notebooks to inspect results, reproduce plots, or tune hyperparameters.

## Contributing

Feel free to open issues or pull requests. For major changes (new features or refactors), please open an issue first to discuss.

## Contact

For questions, reach out to the project owner.