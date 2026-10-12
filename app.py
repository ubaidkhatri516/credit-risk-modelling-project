from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from credit_risk import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_sample_data,
    prepare_features,
    train_and_evaluate,
    explain_prediction,
)

st.set_page_config(page_title="CreditLens | Credit Risk Modeling", page_icon="💳", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
      [data-testid="stMetric"] {background:rgba(100,120,150,.09);border:1px solid rgba(128,128,128,.2);padding:15px;border-radius:13px;}
      .hero {padding:1.2rem 1.4rem;border-radius:18px;background:linear-gradient(115deg,rgba(68,112,255,.16),rgba(31,180,143,.10));border:1px solid rgba(128,128,128,.18);margin-bottom:1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1 style="margin:0">💳 CreditLens</h1>
      <p style="margin:.4rem 0 0 0">Credit Risk Modeling & Explainable Machine Learning</p>
      <p style="margin:.35rem 0 0 0;opacity:.75">Logistic Regression · Decision Tree · Random Forest · SMOTE · SHAP-style feature contributions</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Educational prototype only. Do not use this model to approve or reject real loan applicants. "
    "The included dataset is synthetic and the model is not validated for real-world lending."
)

@st.cache_data
def get_sample():
    return load_sample_data()

with st.sidebar:
    st.header("⚙️ Model settings")
    data_source = st.radio("Dataset", ["Sample dataset (included)", "Upload your own CSV"])
    uploaded = None
    if data_source == "Upload your own CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        st.caption("CSV must include the target column `loan_status` and compatible feature columns.")
    model_name = st.selectbox("Model", ["Logistic Regression", "Decision Tree", "Random Forest"])
    use_smote = st.checkbox("Use SMOTE for class imbalance", value=True)
    test_size = st.slider("Test set size", 0.15, 0.40, 0.25, 0.05)
    train_button = st.button("Train / refresh model", type="primary", use_container_width=True)

if data_source == "Upload your own CSV" and uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        if TARGET_COLUMN not in df.columns:
            st.error(f"CSV must include a `{TARGET_COLUMN}` column.")
            st.stop()
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                st.warning(f"Missing `{col}`. It will be filled with a default value where possible.")
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
else:
    df = get_sample()

st.subheader("Dataset overview")
a, b, c = st.columns(3)
a.metric("Rows", f"{len(df):,}")
b.metric("Features", f"{len([x for x in FEATURE_COLUMNS if x in df.columns])}")
c.metric("Default rate", f"{(df[TARGET_COLUMN].astype(str).str.lower().isin(['1','default','yes','true']).mean()*100):.1f}%")
st.caption("The sample dataset is synthetically generated for learning and reproducibility.")

with st.expander("Preview dataset"):
    st.dataframe(df.head(25), use_container_width=True, hide_index=True)

# Cache only data-independent computations carefully; train on demand.
if "model_result" not in st.session_state or train_button or "last_config" not in st.session_state or st.session_state.get("last_config") != (model_name, use_smote, test_size, data_source, len(df)):
    try:
        with st.spinner("Training model and calculating evaluation metrics..."):
            st.session_state["model_result"] = train_and_evaluate(
                df=df,
                model_name=model_name,
                use_smote=use_smote,
                test_size=test_size,
            )
            st.session_state["last_config"] = (model_name, use_smote, test_size, data_source, len(df))
    except Exception as exc:
        st.error(f"Model training failed: {exc}")
        st.stop()

result = st.session_state["model_result"]
metrics = result["metrics"]
st.divider()
st.subheader("Model evaluation")
st.caption(f"Model: {model_name} · SMOTE: {'Enabled' if use_smote and result['smote_applied'] else 'Not applied'}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}" if metrics["roc_auc"] is not None else "N/A")
m2.metric("Default precision", f"{metrics['precision']:.3f}")
m3.metric("Default recall", f"{metrics['recall']:.3f}")
m4.metric("F1 score", f"{metrics['f1']:.3f}")

left, right = st.columns(2)
with left:
    st.markdown("### Confusion matrix")
    cm = result["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=["Actual: repaid", "Actual: default"], columns=["Predicted: repaid", "Predicted: default"])
    fig_cm = px.imshow(
        cm_df, text_auto=True, aspect="auto", color_continuous_scale="Blues",
        labels=dict(x="Predicted class", y="Actual class", color="Count"),
    )
    fig_cm.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_cm, use_container_width=True)
with right:
    st.markdown("### Classification report")
    report_df = pd.DataFrame(result["classification_report"]).T
    st.dataframe(report_df.round(3), use_container_width=True)

st.markdown("### Feature importance")
importance = result["feature_importance"].sort_values("importance", ascending=True)
fig_imp = px.bar(
    importance, x="importance", y="feature", orientation="h",
    labels={"importance": "Relative importance", "feature": "Feature"},
    title="Model-level feature importance (not causal effects)",
)
fig_imp.update_layout(margin=dict(t=40, b=10, l=10, r=10))
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("### Explore a hypothetical applicant")
st.caption("Change inputs to see the model score and its feature contributions. This is a classroom demonstration, not a lending decision.")
with st.form("applicant_form"):
    x1, x2, x3 = st.columns(3)
    with x1:
        annual_income = st.number_input("Annual income (₹)", min_value=10000, max_value=10000000, value=600000, step=25000)
        loan_amount = st.number_input("Loan amount (₹)", min_value=10000, max_value=5000000, value=250000, step=10000)
    with x2:
        interest_rate = st.slider("Interest rate (%)", 4.0, 35.0, 13.0, 0.5)
        dti = st.slider("Debt-to-income ratio (%)", 0.0, 80.0, 25.0, 1.0)
    with x3:
        credit_history = st.slider("Credit history length (years)", 0, 40, 6)
        delinquencies = st.slider("Previous delinquencies", 0, 10, 0)
    submitted = st.form_submit_button("Estimate default risk", type="primary")

if submitted:
    applicant = pd.DataFrame([{
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "debt_to_income": dti,
        "credit_history_years": credit_history,
        "delinquencies": delinquencies,
        "employment_years": 3.0,
        "home_ownership": "RENT",
        "loan_purpose": "personal",
    }])
    try:
        explanation = explain_prediction(result["pipeline"], applicant, result["model_name"])
        score = explanation["probability"]
        st.metric("Estimated default probability", f"{score:.1%}")
        st.progress(float(max(0.0, min(1.0, score))))
        st.caption("This probability is produced by a model trained on synthetic educational data. It is not calibrated for real lending.")
        st.markdown("#### Largest contributions to this model score")
        st.dataframe(explanation["contributions"].head(8), use_container_width=True, hide_index=True)
        st.info(explanation["explanation_note"])
    except Exception as exc:
        st.error(f"Could not explain this sample prediction: {exc}")

st.markdown("### Download evaluation artifacts")
eval_summary = pd.DataFrame([{
    "model": result["model_name"],
    "smote_requested": use_smote,
    "smote_applied": result["smote_applied"],
    **metrics,
}])
st.download_button("⬇️ Download metrics CSV", eval_summary.to_csv(index=False), "credit_risk_metrics.csv", "text/csv")
st.download_button("⬇️ Download test predictions CSV", result["predictions"].to_csv(index=False), "credit_risk_test_predictions.csv", "text/csv")

with st.expander("Methodology and limitations"):
    st.markdown(
        """
        - The target is `loan_status`: 1 means default, 0 means repaid.
        - Numeric missing values are median-imputed; categorical missing values are imputed with the most frequent value and one-hot encoded.
        - The data is split into training and test sets before resampling. SMOTE is applied only to the training data to reduce leakage.
        - ROC-AUC measures ranking ability; precision, recall and F1 summarize default-class performance at a 0.5 threshold.
        - Feature importance is model-specific and should not be interpreted as causation.
        - For actual lending use, this project would need representative lawful data, leakage checks, probability calibration, fairness and adverse-impact testing, governance, monitoring, and human review.
        """
    )

st.caption("CreditLens · Educational data analytics project · Not financial advice or a real credit underwriting tool.")
