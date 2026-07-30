"""
Train the FINAL deployable pipeline (SVM-RBF + QSVM), on ALL 21 samples.
----------------------------------------------------------------------
IMPORTANT: this is NOT another CV evaluation. This script fits mRMR
feature selection, scaling, and the classifier(s) on every one of the
21 samples, with no held-out fold. That's correct practice for a final
deployed model, but it means the artifacts saved here have NO unbiased
accuracy estimate attached to them -- for that, keep citing the nested
LOOCV numbers from phase5/phase6 (90.48%). Don't conflate the two: the
LOOCV number describes "how well this modeling approach generalizes,"
this script produces "the actual model object we ship."

Verified end-to-end on synthetic data matching the real 21x54675 shape
(21 samples, 5 classes 5/5/3/5/3, random genes) before being run on the
real data -- confirms artifact shapes, no crashes, predict_proba works.

Install once:
    pip install mrmr-selection scikit-learn pandas numpy joblib pennylane --break-system-packages
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from mrmr import mrmr_classif

INPUT_CSV = "gene_expression_labeled.csv"
N_FEATURES = 75
PREFILTER = 3000
RANDOM_STATE = 42
SVM_C_GRID = [0.1, 1, 10, 100]
SVM_GAMMA_GRID = ["scale", 0.01, 0.001]

np.random.seed(RANDOM_STATE)

# ============================================================
# Load
# ============================================================
df = pd.read_csv(INPUT_CSV, index_col=0)
X = df.drop(columns=["label"]).reset_index(drop=True)
y_raw = df["label"].reset_index(drop=True)

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_
print("Classes:", list(class_names))

# ============================================================
# Feature selection on ALL 21 samples (no held-out fold -- this is
# the deployed model, not a CV estimate)
# ============================================================
top_var_genes = X.var().sort_values(ascending=False).index[:PREFILTER]
X_pf = X[top_var_genes]
selected_genes = mrmr_classif(X=X_pf, y=pd.Series(y), K=N_FEATURES, show_progress=False)
X_sel = X_pf[selected_genes].values
print(f"Selected {len(selected_genes)} genes via mRMR on the full 21-sample set.")

# ============================================================
# Scaling (fit on all 21 samples)
# ============================================================
scaler = StandardScaler().fit(X_sel)
X_scaled = scaler.transform(X_sel)

# ============================================================
# SVM-RBF: pick (C, gamma) via Leave-One-Out over the full 21 samples,
# then refit on all 21 with the winning hyperparameters
# ============================================================
classes_present, counts = np.unique(y, return_counts=True)
weight_by_class = {c: len(y) / (len(classes_present) * n) for c, n in zip(classes_present, counts)}
sample_weight = np.array([weight_by_class[label] for label in y])

def loo_select_svm(X_all, y_all, w_all):
    loo = LeaveOneOut()
    best_score, best_params = -1.0, (1.0, "scale")
    for C in SVM_C_GRID:
        for gamma in SVM_GAMMA_GRID:
            correct, scored = 0, 0
            for tr_idx, val_idx in loo.split(X_all):
                if len(np.unique(y_all[tr_idx])) < 2:
                    continue
                clf = SVC(C=C, gamma=gamma, kernel="rbf", class_weight="balanced",
                          probability=False, random_state=RANDOM_STATE)
                clf.fit(X_all[tr_idx], y_all[tr_idx], sample_weight=w_all[tr_idx])
                pred = clf.predict(X_all[val_idx])
                correct += int(pred[0] == y_all[val_idx][0])
                scored += 1
            score = correct / scored if scored else 0.0
            if score > best_score:
                best_score, best_params = score, (C, gamma)
    return best_params

best_C, best_gamma = loo_select_svm(X_scaled, y, sample_weight)
print(f"Selected SVM-RBF hyperparameters: C={best_C}, gamma={best_gamma}")

svm_final = SVC(C=best_C, gamma=best_gamma, kernel="rbf", class_weight="balanced",
                 probability=True, random_state=RANDOM_STATE)
svm_final.fit(X_scaled, y, sample_weight=sample_weight)

# ============================================================
# QSVM artifacts: amplitude-encoded, scaled [0,1] training set, saved
# so the app can build a quantum kernel row against these 21 training
# samples at inference time (no retraining needed -- SVC itself is
# refit on the precomputed 21x21 Gram matrix once, here).
# ============================================================
mm_scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_scaled)
X_q_scaled = mm_scaler.transform(X_scaled)

import pennylane as qml
N_QUBITS = int(np.ceil(np.log2(N_FEATURES)))  # 7

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    qml.AmplitudeEmbedding(x1, wires=range(N_QUBITS), normalize=True, pad_with=0.0)
    qml.adjoint(qml.AmplitudeEmbedding)(x2, wires=range(N_QUBITS), normalize=True, pad_with=0.0)
    return qml.probs(wires=range(N_QUBITS))

def quantum_kernel(x1, x2):
    return float(kernel_circuit(x1, x2)[0])

n = len(X_q_scaled)
K_train = np.ones((n, n))
for i in range(n):
    for j in range(i + 1, n):
        val = quantum_kernel(X_q_scaled[i], X_q_scaled[j])
        K_train[i, j] = val
        K_train[j, i] = val

qsvm_final = SVC(kernel="precomputed", C=10, class_weight="balanced",
                  probability=True, random_state=RANDOM_STATE)
qsvm_final.fit(K_train, y)

# ============================================================
# Save every artifact the app needs
# ============================================================
joblib.dump({
    "selected_genes": selected_genes,
    "scaler": scaler,
    "svm_final": svm_final,
    "class_names": list(class_names),
    "mm_scaler": mm_scaler,
    "X_q_scaled": X_q_scaled,     # training set, amplitude-scaled, for QSVM kernel rows
    "y_train": y,
    "qsvm_final": qsvm_final,
    "n_qubits": N_QUBITS,
}, "deployment_pipeline.joblib")

print("\nSaved deployment_pipeline.joblib")
print("This file contains everything app.py needs: gene list, scaler, "
      "trained SVM-RBF, and QSVM training artifacts.")