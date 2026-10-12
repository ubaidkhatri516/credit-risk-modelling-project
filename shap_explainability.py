"""
Optional SHAP example.

This script trains a Random Forest on the included synthetic dataset and prints a
SHAP summary. Install requirements first. For larger datasets, consider sampling
the training data to control runtime.
"""
import matplotlib.pyplot as plt
import shap

from credit_risk import load_sample_data, prepare_features, _build_preprocessor, _build_model
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df = load_sample_data()
X, y = prepare_features(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

preprocessor = _build_preprocessor()
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Apply SMOTE to the training split only.
minority_count = int(y_train.value_counts().min())
if minority_count >= 2:
    X_train_transformed, y_train = SMOTE(
        random_state=42, k_neighbors=min(5, minority_count - 1)
    ).fit_resample(X_train_transformed, y_train)

model = _build_model("Random Forest")
model.fit(X_train_transformed, y_train)

feature_names = preprocessor.get_feature_names_out()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_transformed)

# SHAP's output shape varies by SHAP version and model; select the default class when needed.
if isinstance(shap_values, list):
    values_for_default = shap_values[1]
else:
    import numpy as np
    values_array = np.asarray(shap_values)
    if values_array.ndim == 3:
        values_for_default = values_array[:, :, 1]
    else:
        values_for_default = values_array

shap.summary_plot(
    values_for_default,
    X_test_transformed,
    feature_names=feature_names,
    show=False,
)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=160, bbox_inches="tight")
print("Saved SHAP summary plot to shap_summary.png")
