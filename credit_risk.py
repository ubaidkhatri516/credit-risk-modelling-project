"""Data preparation, model training, metrics, and local model explanations."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = PROJECT_DIR / "data" / "credit_risk_sample.csv"

TARGET_COLUMN = "loan_status"
NUMERIC_FEATURES = [
    "annual_income",
    "loan_amount",
    "interest_rate",
    "debt_to_income",
    "credit_history_years",
    "delinquencies",
    "employment_years",
]
CATEGORICAL_FEATURES = ["home_ownership", "loan_purpose"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_sample_data() -> pd.DataFrame:
    """Load the included synthetic dataset."""
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Sample dataset not found: {SAMPLE_PATH}")
    return pd.read_csv(SAMPLE_PATH)


def normalize_target(series: pd.Series) -> pd.Series:
    """Convert common binary target representations to 0/1."""
    if pd.api.types.is_numeric_dtype(series):
        values = set(series.dropna().unique())
        if values.issubset({0, 1}):
            return series.astype(int)
    mapping = {
        "0": 0, "repaid": 0, "fully paid": 0, "paid": 0, "no": 0, "false": 0,
        "1": 1, "default": 1, "charged off": 1, "late": 1, "yes": 1, "true": 1,
    }
    normalized = series.astype(str).str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        bad = series[normalized.isna()].astype(str).unique()[:5]
        raise ValueError(
            f"Target `{TARGET_COLUMN}` must be binary (0=non-default, 1=default). "
            f"Unrecognized examples: {list(bad)}"
        )
    return normalized.astype(int)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select expected features, fill absent columns with defaults, and normalize target."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must include target column `{TARGET_COLUMN}`.")
    working = df.copy()
    y = normalize_target(working[TARGET_COLUMN])
    X = working.reindex(columns=FEATURE_COLUMNS).copy()

    # Helpful defaults let users experiment with a dataset that has some missing columns,
    # but a warning in the UI reminds them that external schemas should be mapped carefully.
    numeric_defaults = {
        "annual_income": 500000,
        "loan_amount": 200000,
        "interest_rate": 12.0,
        "debt_to_income": 25.0,
        "credit_history_years": 5,
        "delinquencies": 0,
        "employment_years": 3,
    }
    categorical_defaults = {"home_ownership": "RENT", "loan_purpose": "personal"}
    for col, default in numeric_defaults.items():
        X[col] = pd.to_numeric(X[col], errors="coerce")
        if X[col].isna().all():
            X[col] = default
    for col, default in categorical_defaults.items():
        X[col] = X[col].fillna(default).astype(str)
    return X, y


def _build_preprocessor():
    numeric_pipe = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _build_model(model_name: str):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, class_weight=None, random_state=42)
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=5, random_state=42, class_weight="balanced"
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=250, max_depth=8, min_samples_leaf=3, random_state=42,
            class_weight=None, n_jobs=-1
        )
    raise ValueError(f"Unknown model: {model_name}")


def train_and_evaluate(
    df: pd.DataFrame,
    model_name: str = "Logistic Regression",
    use_smote: bool = True,
    test_size: float = 0.25,
) -> dict:
    """Split, preprocess, optionally SMOTE-resample training data, fit model, and evaluate."""
    X, y = prepare_features(df)
    if y.nunique() < 2:
        raise ValueError("The target must contain both classes (0=repaid and 1=default).")
    if y.value_counts().min() < 3:
        raise ValueError("At least three rows from each class are needed for a reliable split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    preprocessor = _build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    smote_applied = False
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            # SMOTE works on the transformed numeric matrix; one-hot features may become
            # fractional. For a teaching demo this illustrates resampling, but SMOTENC or
            # a categorical-aware workflow may be preferable for production tabular data.
            minority_count = int(y_train.value_counts().min())
            if minority_count >= 2 and y_train.nunique() == 2:
                smote = SMOTE(random_state=42, k_neighbors=min(5, minority_count - 1))
                X_fit, y_fit = smote.fit_resample(X_train_processed, y_train)
                smote_applied = True
            else:
                X_fit, y_fit = X_train_processed, y_train
        except ImportError:
            warnings.warn("imbalanced-learn is not installed; continuing without SMOTE.")
            X_fit, y_fit = X_train_processed, y_train
    else:
        X_fit, y_fit = X_train_processed, y_train

    model = _build_model(model_name)
    model.fit(X_fit, y_fit)

    predictions = model.predict(X_test_processed)
    probabilities = model.predict_proba(X_test_processed)[:, 1]
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(
        y_test, predictions, labels=[0, 1], target_names=["Repaid", "Default"],
        output_dict=True, zero_division=0
    )
    try:
        auc = float(roc_auc_score(y_test, probabilities))
    except ValueError:
        auc = None

    metrics = {
        "roc_auc": auc,
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feature_{i}" for i in range(X_train_processed.shape[1])])

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": np.asarray(importances, dtype=float)}
    ).sort_values("importance", ascending=False)

    prediction_df = X_test.reset_index(drop=True).copy()
    prediction_df["actual_default"] = y_test.reset_index(drop=True)
    prediction_df["predicted_default"] = predictions
    prediction_df["default_probability"] = probabilities

    pipeline = {
        "preprocessor": preprocessor,
        "model": model,
        "feature_names": list(feature_names),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }

    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_importance": feature_importance,
        "predictions": prediction_df,
        "smote_applied": smote_applied,
        "test_y": y_test.reset_index(drop=True),
    }


def explain_prediction(pipeline: dict, applicant: pd.DataFrame, model_name: str) -> dict:
    """
    Return a local contribution approximation.

    For logistic regression, contributions are coefficient * standardized encoded input,
    centered on the intercept. For tree models, report feature perturbation effects by
    replacing one transformed feature with its training reference value where possible.
    This is deliberately labelled as a lightweight local explanation, not a substitute
    for SHAP/LIME. The standalone explainability notebook demonstrates SHAP when installed.
    """
    preprocessor = pipeline["preprocessor"]
    model = pipeline["model"]
    transformed = preprocessor.transform(applicant)
    if hasattr(transformed, "toarray"):
        transformed_array = transformed.toarray()
    else:
        transformed_array = np.asarray(transformed)
    feature_names = pipeline["feature_names"]
    probability = float(model.predict_proba(transformed)[:, 1][0])

    contributions = []
    if hasattr(model, "coef_"):
        coefs = np.asarray(model.coef_[0])
        raw = coefs * transformed_array[0]
        for name, value in zip(feature_names, raw):
            contributions.append({"feature": str(name), "contribution": float(value)})
        note = (
            "For logistic regression, values are coefficient × encoded feature values on the "
            "model's linear score scale. They are not causal effects and are not exact probability-point changes."
        )
    else:
        baseline = np.zeros(transformed_array.shape[1], dtype=float)
        baseline_probability = float(model.predict_proba([baseline])[:, 1][0])
        for idx, name in enumerate(feature_names):
            perturbed = transformed_array.copy()
            perturbed[0, idx] = baseline[idx]
            changed_probability = float(model.predict_proba(perturbed)[:, 1][0])
            contributions.append(
                {"feature": str(name), "contribution": probability - changed_probability}
            )
        note = (
            "For tree models, contributions are a simple one-feature perturbation approximation relative "
            "to a zero-vector baseline. This is not SHAP or LIME and can be misleading with correlated features."
        )

    contribution_df = pd.DataFrame(contributions)
    if not contribution_df.empty:
        contribution_df["absolute_contribution"] = contribution_df["contribution"].abs()
        contribution_df = contribution_df.sort_values("absolute_contribution", ascending=False)
    return {
        "probability": probability,
        "contributions": contribution_df,
        "explanation_note": note,
    }
