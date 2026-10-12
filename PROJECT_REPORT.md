# Project Report: Credit Risk Modeling

## 1. Introduction

Credit risk modeling estimates the likelihood that a borrower will fail to repay a loan. Lenders use risk models to support portfolio monitoring and underwriting, but any real-world use requires representative data, validation, governance, fairness analysis, and compliance with applicable laws. This student project demonstrates the machine learning workflow on a synthetic dataset.

## 2. Problem statement

Loan datasets may contain more repaid loans than defaults. A model trained without considering this class imbalance may under-detect the minority default class. The project compares common classification algorithms, applies SMOTE to the training set when enabled, and evaluates performance using several metrics.

## 3. Objectives

1. Prepare loan-applicant data for classification.
2. Handle missing values and encode categorical features.
3. Split data into stratified training and test sets.
4. Address class imbalance using SMOTE on training data only.
5. Train Logistic Regression, Decision Tree, and Random Forest models.
6. Evaluate models using ROC-AUC, precision, recall, F1, and a confusion matrix.
7. Explain model behavior using feature importance and a SHAP example.
8. Provide an interactive Streamlit dashboard.

## 4. Dataset

The project includes a synthetic dataset generated for demonstration and repeatability. It includes income, loan amount, interest rate, debt-to-income ratio, credit history length, delinquencies, employment history, home ownership, loan purpose, and a binary outcome.

The target is `loan_status`, where 0 represents repaid/non-default and 1 represents default. The dataset is not actual LendingClub data and should not be represented as real borrower records. A real LendingClub dataset may require careful status mapping and removal of leakage-prone fields.

## 5. Methodology

### 5.1 Data preprocessing

- Missing numeric values are imputed with the median.
- Missing categorical values are imputed with the most frequent category.
- Numeric columns are standardized.
- Categorical columns are one-hot encoded.
- The target is normalized to 0/1.

### 5.2 Train/test split

A stratified split is used to preserve the approximate target class ratio in training and test data. The test split is held out from SMOTE and model fitting.

### 5.3 SMOTE

Synthetic Minority Over-sampling Technique (SMOTE) creates synthetic minority-class examples using nearest-neighbor interpolation. In this project it is applied only to the transformed training set. It must not be applied before splitting, because that can leak related synthetic observations into both training and test sets.

Because one-hot encoded categories are not naturally continuous, plain SMOTE can produce fractional encoded values. A categorical-aware method such as SMOTENC should be considered for robust mixed-type tabular modeling.

### 5.4 Models

- **Logistic Regression:** linear probabilistic baseline for binary classification.
- **Decision Tree:** interpretable non-linear model with feature-based splits.
- **Random Forest:** ensemble of decision trees intended to improve robustness.

### 5.5 Evaluation metrics

| Metric | Meaning |
|---|---|
| ROC-AUC | Ability to rank default cases above non-default cases |
| Precision | Fraction of predicted defaults that are actual defaults |
| Recall | Fraction of actual defaults detected |
| F1 score | Harmonic mean of precision and recall |
| Confusion matrix | Counts of correct and incorrect predictions |

In this project, a probability threshold of 0.5 is used for the displayed classification metrics. This is a demonstration default, not a recommended lending threshold.

### 5.6 Explainability

The dashboard shows model-level feature importance and a simple local contribution approximation. The separate SHAP module shows how to calculate SHAP values for a tree-based model. SHAP values describe contributions relative to a defined background/reference distribution; they do not prove causation or fairness.

## 6. System architecture

```text
Synthetic dataset / user CSV
           |
           v
      Data validation
           |
           v
 Imputation + scaling + encoding
           |
           v
    Stratified train/test split
           |
           v
       SMOTE on training data
           |
           v
 Logistic Regression / Decision Tree / Random Forest
           |
           v
 Metrics + confusion matrix + feature explanations
           |
           v
     Streamlit dashboard + CSV exports
```

## 7. Results

The dashboard calculates model metrics at runtime based on the selected model, resampling option, split size, and dataset. The exact numerical results are intentionally not hard-coded in this report. Record the metrics shown by your own run before submitting, and state that they come from the synthetic dataset.

## 8. Limitations

1. Synthetic data cannot establish real-world predictive performance.
2. Class imbalance treatment may alter precision and probability calibration.
3. Plain SMOTE on one-hot-encoded data may create fractional categorical representations.
4. The demonstration does not perform probability calibration or threshold optimization.
5. Feature importance and SHAP explanations do not establish causal relationships.
6. No formal fairness audit, adverse-impact analysis, or regulatory review is performed.
7. Real lending datasets may contain missing values, target leakage, historical policy effects, and sensitive or legally restricted attributes.

## 9. Future scope

- Use a documented, authorized lending dataset such as a properly prepared historical LendingClub release.
- Compare baseline training with SMOTE and class-weighted models.
- Tune the decision threshold based on validated error costs.
- Add calibration curves, precision-recall curves, and cross-validation.
- Add SHAP summary and local waterfall plots.
- Conduct subgroup performance and fairness analysis with lawful, carefully governed data.
- Add monitoring for data drift and model performance.

## 10. Conclusion

This project demonstrates a standard credit-risk classification pipeline, including preprocessing, imbalance handling, model comparison, evaluation, and explainability. It is suitable as an educational prototype and starting point for further research. It is not a validated credit scoring system and should not be used for real lending decisions.

## 11. References

1. scikit-learn documentation: https://scikit-learn.org/stable/
2. imbalanced-learn SMOTE: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
3. SHAP documentation: https://shap.readthedocs.io/
4. Streamlit documentation: https://docs.streamlit.io/
5. LendingClub historical data information: https://www.lendingclub.com/info/download-data.action
