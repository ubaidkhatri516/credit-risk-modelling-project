# CreditLens — Credit Risk Modeling

A data analytics / machine learning project that estimates loan default risk using Logistic Regression, Decision Tree, or Random Forest. It demonstrates class imbalance handling with SMOTE, evaluation metrics, and explainability-oriented feature analysis in a Streamlit dashboard.

> **Educational use only:** the included dataset is synthetic. This project is not suitable for making real credit decisions and must not be used to approve or reject real loan applicants.

## Features

- Included synthetic dataset; no download required to run.
- Optional upload of a CSV with `loan_status` target.
- Three model choices: Logistic Regression, Decision Tree, Random Forest.
- Stratified train/test split.
- Numeric imputation, scaling, and categorical one-hot encoding.
- Optional SMOTE resampling on training data only.
- Evaluation metrics: ROC-AUC, default-class precision, recall, F1, confusion matrix, classification report.
- Model-level feature importance.
- Hypothetical applicant scoring demonstration.
- Downloadable metrics and test predictions.

## Project files

```text
credit_risk_modeling_project/
├── app.py
├── credit_risk.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── data/
│   └── credit_risk_sample.csv
└── tests/
    └── test_credit_risk.py
```

## Setup on Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

If `streamlit` is not recognized:

```bash
python -m streamlit run app.py
```

Choose **Sample dataset (included)** to start. No API keys are needed.

## Dataset format

The included data is synthetic and contains:

- `annual_income`
- `loan_amount`
- `interest_rate`
- `debt_to_income`
- `credit_history_years`
- `delinquencies`
- `employment_years`
- `home_ownership`
- `loan_purpose`
- `loan_status` — target: `0` = repaid, `1` = default

If using another dataset, map its columns to this schema before training. Do not assume a LendingClub dataset uses the same labels or that every status can safely be mapped to default/non-default without reviewing the data dictionary.

## SMOTE and data leakage

The code splits data first, fits preprocessing on the training split, and applies SMOTE only to the transformed training data. The test set remains untouched. This prevents resampled examples from leaking into the test set. Because one-hot encoded categorical features can become fractional under ordinary SMOTE, a production tabular workflow should consider `SMOTENC` or other categorical-aware approaches.

## Evaluation

- **ROC-AUC:** how well the model ranks default cases above non-default cases.
- **Precision:** among predicted defaults, the proportion that were actual defaults.
- **Recall:** among actual defaults, the proportion detected by the model.
- **F1:** harmonic mean of precision and recall.
- **Confusion matrix:** counts of correct and incorrect classifications.

The classification threshold is 0.5 in this demonstration. The best threshold depends on validated business objectives and the relative costs of false positives and false negatives.

## Explainability note

The dashboard shows model-level feature importance and local contribution approximations. The separate `shap_explainability.py` module provides a SHAP TreeExplainer example for a trained tree model. The dashboard's lightweight contribution view is **not a SHAP or LIME explanation**; use the dedicated SHAP module for SHAP values and clearly document the method used in any report.

## Tests

```bash
pytest -q
```

## Upload to GitHub

1. Extract the ZIP.
2. Create a new repository at https://github.com/new.
3. Open a terminal in the extracted folder and run:

```bash
git init
git add .
git commit -m "Add credit risk modeling project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the URL with your repository URL.

## References

- scikit-learn: https://scikit-learn.org/stable/
- imbalanced-learn SMOTE: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
- SHAP documentation: https://shap.readthedocs.io/
- Streamlit: https://docs.streamlit.io/
- LendingClub historical data information: https://www.lendingclub.com/info/download-data.action
