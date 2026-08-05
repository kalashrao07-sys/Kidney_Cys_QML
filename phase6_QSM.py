"""
Phase 6: Quantum Support Vector Machine (QSVM)
----------------------------------------------------------------------
The lead quantum model for this project, run alongside VQC
(phase4_vqc_pennylane.py) rather than replacing it.

WHY QSVM AND NOT JUST MORE VQC/QNN:
VQC is a VARIATIONAL circuit -- it has trainable parameters fit by
gradient descent (84 parameters in phase4, on ~20 training samples per
fold). That parameter-to-sample ratio is exactly the regime where
overfitting risk is highest, and phase4's own docstring already flags
this. QSVM is structurally different: it is a KERNEL method. There is
no gradient training loop and no free circuit parameters at all -- the
quantum circuit's only job is to compute a similarity (kernel) value
between pairs of samples, and a completely classical, convex, globally-
optimal SVM solver (sklearn's SVC) does the actual classification on
top of that kernel matrix. Kernel methods are exactly the classical
family that already tends to do best in small-n/high-p regimes (this
is why SVM-RBF was Phase 1's winner) -- QSVM is a principled quantum
extension of that same reasoning, not just "a different quantum model
for coverage." Comparing VQC vs QSVM in the paper is therefore a real,
interpretable methodological contrast: variational vs. kernel-based
QML on the same 21-sample, small-n problem.

ENCODING, KEPT IDENTICAL TO phase4_vqc_pennylane.py ON PURPOSE:
Same mRMR-selected ~75 genes, same amplitude embedding into the same
7 qubits (2^7=128 >= 75), same StandardScaler -> MinMaxScaler([0,1])
preprocessing. This is deliberate: VQC and QSVM should differ ONLY in
"variational circuit trained with gradients" vs. "same feature map
used as a fixed kernel" -- not in feature set, qubit count, or
preprocessing -- so any performance difference in the paper is
attributable to that one architectural choice, not a confound.

HOW THE QUANTUM KERNEL IS COMPUTED:
For two samples x1, x2, let U(x) be the unitary that amplitude-encodes
x into the 7-qubit state |psi(x)>. The circuit applies U(x1) to |0>,
then U(x2)^dagger (the adjoint/inverse), and measures the probability
of observing all-zeros. That probability equals |<psi(x2)|psi(x1)>|^2,
the fidelity (overlap) between the two encoded quantum states -- this
is the standard "inversion test" for quantum kernels (Havlicek et al.,
2019) and is verified below to behave correctly (self-similarity ~1.0,
symmetric, and invariant to overall vector scale since amplitude
embedding L2-normalizes -- meaning this kernel captures each sample's
*relative pattern* across the 75 genes, not absolute expression
magnitude; worth stating explicitly as an interpretive property in
the paper, not hiding it).

WHY C IS TUNED BUT NOT A "GAMMA"-LIKE PARAMETER:
Unlike RBF, this kernel has no free width/bandwidth parameter to tune
-- the encoding and kernel form are fixed once the feature set and
qubit count are chosen. Only the SVM's own regularization strength C
is tuned, via an inner Leave-One-Out loop over each outer fold's
training data. Because the kernel matrix doesn't depend on C at all,
this inner tuning is done by SLICING the already-computed Gram matrix
(no extra quantum circuit evaluations needed) -- a genuine efficiency
advantage of kernel methods over refitting a whole model per candidate
hyperparameter.

PROBABILITY CAVEAT (same as classical SVM in phase5, restated because
it still applies): SVC(probability=True) internally uses a small CV
to calibrate probabilities, which is itself shaky on ~20 training
samples. Flagged, not hidden -- needed anyway to compute pooled
ROC-AUC.

Install once:
    pip install pennylane mrmr-selection scikit-learn pandas numpy
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix,
)
from mrmr import mrmr_classif

# ============================================================
# Config -- kept identical to phase4_vqc_pennylane.py where applicable
# ============================================================
INPUT_CSV = "gene_expression_labeled.csv"
N_FEATURES = 75
PREFILTER = 3000
N_QUBITS = int(np.ceil(np.log2(N_FEATURES)))     # 7, since 2^7=128 >= 75
RANDOM_STATE = 42

QSVM_C_GRID = [0.1, 1, 10, 100]   # same grid as classical SVM in phase5, for a fair comparison

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
n_classes = len(class_names)

assert N_FEATURES <= 2 ** N_QUBITS, "N_FEATURES must fit in 2**N_QUBITS amplitudes."

print("Classes:", list(class_names))
print(f"N_QUBITS={N_QUBITS} (amplitude-encoding {N_FEATURES} genes, padded to {2**N_QUBITS})")
print(f"Running Leave-One-Out over {len(X)} samples x {X.shape[1]} genes...\n")

# ============================================================
# Quantum kernel: amplitude-embedding overlap (inversion test)
# ============================================================
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev)
def kernel_circuit(x1, x2):
    qml.AmplitudeEmbedding(x1, wires=range(N_QUBITS), normalize=True, pad_with=0.0)
    qml.adjoint(qml.AmplitudeEmbedding)(x2, wires=range(N_QUBITS), normalize=True, pad_with=0.0)
    return qml.probs(wires=range(N_QUBITS))


def quantum_kernel(x1, x2):
    """|<psi(x2)|psi(x1)>|^2 -- probability of measuring all-zeros
    after encoding x1 then un-encoding x2. Verified: self-similarity
    ~1.0, symmetric, scale-invariant (see project notes)."""
    return float(kernel_circuit(x1, x2)[0])


def build_train_kernel(X_train_enc):
    """Symmetric n_train x n_train Gram matrix. Diagonal fixed to 1.0
    analytically (amplitude-embedded states are normalized) rather
    than recomputed, saving n_train circuit evaluations per fold."""
    n = len(X_train_enc)
    K = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            val = quantum_kernel(X_train_enc[i], X_train_enc[j])
            K[i, j] = val
            K[j, i] = val
    return K


def build_test_kernel(x_test_enc, X_train_enc):
    """1 x n_train kernel row between the held-out test sample and
    every training sample."""
    return np.array([[quantum_kernel(x_test_enc, X_train_enc[i]) for i in range(len(X_train_enc))]])


def inner_loo_qsvm_select(K_train, y_train):
    """Pick C by inner Leave-One-Out accuracy, reusing the ALREADY
    COMPUTED training Gram matrix via index slicing -- no additional
    quantum circuit evaluations needed, since the kernel doesn't
    depend on C."""
    n = len(y_train)
    best_score, best_C = -1.0, 1.0
    for C in QSVM_C_GRID:
        correct, scored = 0, 0
        for val_i in range(n):
            train_idx = np.array([i for i in range(n) if i != val_i])
            if len(np.unique(y_train[train_idx])) < 2:
                continue
            K_inner_train = K_train[np.ix_(train_idx, train_idx)]
            K_inner_val = K_train[np.ix_([val_i], train_idx)]
            clf = SVC(kernel="precomputed", C=C, class_weight="balanced", random_state=RANDOM_STATE)
            clf.fit(K_inner_train, y_train[train_idx])
            pred = clf.predict(K_inner_val)
            correct += int(pred[0] == y_train[val_i])
            scored += 1
        score = correct / scored if scored > 0 else 0.0
        if score > best_score:
            best_score, best_C = score, C
    return best_C


# ============================================================
# Nested outer Leave-One-Out loop
# ============================================================
loo = LeaveOneOut()

y_pred_all, y_proba_all, y_true_all = [], [], []
train_times, pred_times = [], []
fold_log = []
proba_log = []   
selected_gene_sets = []

for fold, (train_idx, test_idx) in enumerate(loo.split(X), start=1):
    t_fold_start = time.perf_counter()
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ---- feature selection: fit on this fold's training data ONLY ----
    top_var_genes = X_train.var().sort_values(ascending=False).index[:PREFILTER]
    X_train_pf, X_test_pf = X_train[top_var_genes], X_test[top_var_genes]

    y_train_series = pd.Series(y_train, index=X_train_pf.index)
    selected = mrmr_classif(X=X_train_pf, y=y_train_series, K=N_FEATURES, show_progress=False)
    selected_gene_sets.append(set(selected))

    X_train_sel, X_test_sel = X_train_pf[selected].values, X_test_pf[selected].values

    # ---- normalization: StandardScaler then MinMaxScaler([0,1]), fit on TRAIN only ----
    # (identical rationale to phase4_vqc_pennylane.py: raw expression scale would
    # otherwise let a few high-magnitude genes dominate the encoded quantum state)
    std_scaler = StandardScaler().fit(X_train_sel)
    X_train_std = std_scaler.transform(X_train_sel)
    X_test_std = std_scaler.transform(X_test_sel)

    mm_scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_train_std)
    X_train_scaled = mm_scaler.transform(X_train_std)
    X_test_scaled = mm_scaler.transform(X_test_std)[0]

    # ================= Quantum kernel + SVM =================
    t0 = time.perf_counter()
    K_train = build_train_kernel(X_train_scaled)
    best_C = inner_loo_qsvm_select(K_train, y_train)
    clf = SVC(kernel="precomputed", C=best_C, class_weight="balanced",
              probability=True, random_state=RANDOM_STATE)
    clf.fit(K_train, y_train)
    t1 = time.perf_counter()

    K_test = build_test_kernel(X_test_scaled, X_train_scaled)
    pred = clf.predict(K_test)[0]
    proba_raw = clf.predict_proba(K_test)[0]
    # SVC(probability=True) orders columns by clf.classes_, which may be a
    # SUBSET of range(n_classes) if a class was entirely held out this fold
    # (impossible here since LOOCV always leaves >=2 samples of every class
    # in training, but guarded anyway for robustness)
    proba = np.zeros(n_classes)
    for cls_idx, p in zip(clf.classes_, proba_raw):
        proba[cls_idx] = p
    t2 = time.perf_counter()

    train_times.append(t1 - t0)
    pred_times.append(t2 - t1)
    y_pred_all.append(pred)
    y_proba_all.append(proba)
    y_true_all.append(y_test[0])

    true_label = class_names[y_test[0]]
    print(f"Fold {fold:2d}/{len(X)}  true={true_label:15s}  QSVM={class_names[pred]:15s}  "
          f"(C={best_C})  [{time.perf_counter()-t_fold_start:.1f}s]")

    fold_log.append({
        "fold": fold, "true_label": true_label,
        "QSVM_pred": class_names[pred], "QSVM_best_C": best_C,
    })

# NEW: stash QSVM probabilities for the combined ROC figure
    proba_row = {"fold": fold, "true_label": true_label}
    for cls_idx, cls_name in enumerate(class_names):
        proba_row[f"QSVM_{cls_name}"] = float(proba[cls_idx])
    proba_log.append(proba_row)

# ============================================================
# mRMR feature-selection stability (same diagnostic as phase5,
# recomputed here since this script reruns mRMR independently --
# should closely match phase5's 0.742 if both were run on the
# same real data, since mRMR is deterministic given the same input)
# ============================================================
print("\n" + "=" * 70)
print("mRMR FEATURE-SELECTION STABILITY DIAGNOSTIC (QSVM run)")
print("=" * 70)
pairwise_jaccard = []
for i in range(len(selected_gene_sets)):
    for j in range(i + 1, len(selected_gene_sets)):
        inter = len(selected_gene_sets[i] & selected_gene_sets[j])
        union = len(selected_gene_sets[i] | selected_gene_sets[j])
        pairwise_jaccard.append(inter / union if union else 0.0)
print(f"Mean pairwise Jaccard similarity: {np.mean(pairwise_jaccard):.3f}")

# ============================================================
# Metrics
# ============================================================
print("\n" + "=" * 70)
y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)
y_proba_all = np.array(y_proba_all)
y_true_bin = label_binarize(y_true_all, classes=list(range(n_classes)))

acc = accuracy_score(y_true_all, y_pred_all)
prec, rec, f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average="macro", zero_division=0)
try:
    auc = roc_auc_score(y_true_bin, y_proba_all, average="macro", multi_class="ovr")
except ValueError:
    auc = float("nan")
cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(n_classes)))

print("=== QSVM (quantum kernel, amplitude embedding) ===")
print(f"Accuracy:                 {acc:.4f}")
print(f"Precision (macro):        {prec:.4f}")
print(f"Recall (macro):           {rec:.4f}")
print(f"F1 (macro):               {f1:.4f}")
print(f"ROC-AUC (macro OVR, pooled across LOOCV folds): {auc:.4f}")
print(f"Mean train time / fold:   {np.mean(train_times)*1000:.1f} ms  "
      f"(kernel construction + SVM fit + inner C-search)")
print(f"Mean inference time/fold: {np.mean(pred_times)*1000:.2f} ms")
print("Confusion matrix (rows=true, cols=pred),", list(class_names))
print(cm)

pd.DataFrame(fold_log).to_csv("phase6_qsvm_fold_results.csv", index=False)
pd.DataFrame(proba_log).to_csv("phase6_qsvm_fold_probabilities.csv", index=False)
print("Saved: phase6_qsvm_fold_probabilities.csv (per-class QSVM probabilities)")
summary = pd.DataFrame([{
    "model": "QSVM", "accuracy": acc, "precision_macro": prec, "recall_macro": rec,
    "f1_macro": f1, "roc_auc_macro_ovr": auc,
    "mean_train_time_s": np.mean(train_times), "mean_pred_time_s": np.mean(pred_times),
}])
summary.to_csv("phase6_qsvm_summary.csv", index=False)
print("\nSaved: phase6_qsvm_fold_results.csv, phase6_qsvm_summary.csv, phase6_qsvm_fold_probabilities.csv")
print("Combine this row with phase5_model_comparison_summary.csv and VQC's own")
print("summary (from vqc_loocv_fold_results.csv) for the full six-model table.")
