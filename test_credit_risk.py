import pandas as pd
import pytest

from credit_risk import load_sample_data, normalize_target, prepare_features, train_and_evaluate


def test_sample_dataset_has_expected_schema():
    df = load_sample_data()
    assert "loan_status" in df.columns
    assert len(df) >= 100
    assert set(df["loan_status"].unique()).issubset({0, 1})


def test_normalize_target_labels():
    result = normalize_target(pd.Series(["repaid", "default", "Fully Paid", "Charged Off"]))
    assert result.tolist() == [0, 1, 0, 1]


def test_prepare_features_shapes():
    df = load_sample_data()
    X, y = prepare_features(df)
    assert len(X) == len(y)
    assert "annual_income" in X.columns
    assert set(y.unique()) == {0, 1}


def test_training_returns_metrics():
    df = load_sample_data()
    result = train_and_evaluate(df, model_name="Logistic Regression", use_smote=False)
    assert "roc_auc" in result["metrics"]
    assert "confusion_matrix" in result
    assert result["predictions"].shape[0] > 0
