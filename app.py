"""
Kidney Cyst Gene Expression Classifier -- demo web app
----------------------------------------------------------------------
Loads deployment_pipeline.joblib (built by train_production_pipeline.py)
and classifies a user-uploaded gene expression sample.

INPUT FORMAT REQUIRED (read this before demoing to anyone):
  - A CSV with ONE row per sample, columns = probe IDs (e.g. "1007_s_at",
    "1053_at", ...), matching the same probe naming as
    gene_expression_labeled.csv.
  - Values must already be on the same normalization scale as the
    training data (i.e. RMA-normalized expression, log2 scale). This
    app does NOT run RMA itself -- RMA is a multi-array quantile
    normalization procedure and cannot be meaningfully applied to a
    single new array in isolation. If you need true single-sample
    support from raw .CEL files, that requires implementing frozen RMA
    (fRMA) with parameters frozen from this project's 21 training
    arrays -- flagged as a known limitation, not implemented here.
  - Only the ~75 mRMR-selected genes are actually used for prediction;
    the app will tell you if any are missing from your upload and stop
    rather than silently zero-filling them (zero-filling missing genes
    would quietly corrupt every prediction).

Install once:
    pip install streamlit scikit-learn pandas numpy joblib pennylane --break-system-packages

Run:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Kidney Cyst Gene Expression Classifier", layout="centered")

@st.cache_resource
def load_pipeline():
    return joblib.load("deployment_pipeline.joblib")

pipeline = load_pipeline()
selected_genes = pipeline["selected_genes"]
scaler = pipeline["scaler"]
svm_final = pipeline["svm_final"]
class_names = pipeline["class_names"]

st.title("Kidney Cyst Gene Expression Classifier")
st.caption(
    "Classical (SVM-RBF) and quantum (QSVM) classifiers trained on GSE7869 "
    "(21 samples, 5 classes). Both scored 90.48% under nested Leave-One-Out CV -- "
    "see the paper for the full evaluation. This app runs the FINAL models, "
    "trained on all 21 samples, for actual use, not a repeat of that evaluation."
)

st.warning(
    "Input must be RMA-normalized expression values already on the same scale "
    "as the training data (log2, quantile-normalized), with probe IDs as column "
    "names. This app does not perform RMA itself -- see the module docstring in "
    "app.py for why single-array RMA isn't meaningful without a frozen reference."
)

model_choice = st.radio("Model", ["SVM-RBF (classical, recommended)", "QSVM (quantum kernel)"])

uploaded = st.file_uploader("Upload a CSV: one row per sample, columns = probe IDs", type=["csv"])

if uploaded is not None:
    try:
        user_df = pd.read_csv(uploaded, index_col=0)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    missing = [g for g in selected_genes if g not in user_df.columns]
    if missing:
        st.error(
            f"Your upload is missing {len(missing)} of the {len(selected_genes)} genes "
            f"this model needs (e.g. {missing[:5]}). Prediction refused rather than "
            f"guessed -- please ensure your CSV includes all required probe IDs."
        )
        st.stop()

    X_user = user_df[selected_genes].values
    X_user_scaled = scaler.transform(X_user)

    st.success(f"Loaded {X_user.shape[0]} sample(s), all {len(selected_genes)} required genes present.")

    if st.button("Classify"):
        if model_choice.startswith("SVM"):
            preds = svm_final.predict(X_user_scaled)
            probas = svm_final.predict_proba(X_user_scaled)
        else:
            import pennylane as qml
            mm_scaler = pipeline["mm_scaler"]
            X_q_train = pipeline["X_q_scaled"]
            qsvm_final = pipeline["qsvm_final"]
            n_qubits = pipeline["n_qubits"]

            X_user_q = mm_scaler.transform(X_user_scaled)

            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev)
            def kernel_circuit(x1, x2):
                qml.AmplitudeEmbedding(x1, wires=range(n_qubits), normalize=True, pad_with=0.0)
                qml.adjoint(qml.AmplitudeEmbedding)(x2, wires=range(n_qubits), normalize=True, pad_with=0.0)
                return qml.probs(wires=range(n_qubits))

            def quantum_kernel(x1, x2):
                return float(kernel_circuit(x1, x2)[0])

            with st.spinner("Running quantum kernel circuit against 21 training samples..."):
                K_test = np.array([
                    [quantum_kernel(xq, X_q_train[i]) for i in range(len(X_q_train))]
                    for xq in X_user_q
                ])
            preds = qsvm_final.predict(K_test)
            probas = qsvm_final.predict_proba(K_test)

        for i in range(len(preds)):
            pred_label = class_names[preds[i]]
            st.subheader(f"Sample {i+1}: **{pred_label}**")
            proba_df = pd.DataFrame({
                "Class": class_names,
                "Probability": probas[i],
            }).sort_values("Probability", ascending=False)
            st.bar_chart(proba_df.set_index("Class"))
            st.dataframe(proba_df, hide_index=True)

st.divider()
st.caption(
    "Limitations: n=21 training samples, several likely from the same 5 kidneys "
    "(patient-level leakage risk, not corrected here). Predictions on samples very "
    "different from the training distribution (different platform, tissue prep, "
    "batch) should not be trusted."
)