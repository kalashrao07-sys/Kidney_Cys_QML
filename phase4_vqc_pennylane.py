"""
Phase 4: Deep Variational Quantum Classifier (VQC) in PennyLane
----------------------------------------------------------------------
This is the confirmed "quantum" contribution of the paper (NOT the PQC/
encryption script -- that is optional bonus material only, see
PROJECT_KNOWLEDGE.md). Benchmarked against phase3's classical models
using the *same* nested Leave-One-Out CV, so the only thing that differs
between phase3 and this script is the classifier itself.

ARCHITECTURE -- follows arXiv:2505.02033 ("Quantum-Enhanced Classification
of Brain Tumors Using DNA Microarray Gene Expression Profiles"), whose
setup is structurally almost identical to ours: microarray gene expression,
~tens of thousands of raw genes, small sample size, 5 classes (their 4
tumor types + normal <-> our 4 cyst categories + Normal_Control).
  1. Feature mapping: AMPLITUDE encoding (not angle encoding). This is the
     only encoding that can fit our 75 mRMR-selected genes onto a small
     number of qubits -- angle encoding needs 1 qubit/feature, i.e. 75
     qubits, which is not simulable. Amplitude encoding packs up to 2^n
     features into n qubits, so 75 features fit into 7 qubits (2^7=128).
  2. Two sequential Hardware-Efficient Ansatze (HEA), exactly as
     described in the paper:
       HEA1: Hadamard + parameterized RX,RY per qubit -> CNOT ring +
             Toffoli ring entanglers
       HEA2: Hadamard + parameterized RY,RZ per qubit -> CNOT ring +
             Toffoli ring entanglers
     repeated for N_CYCLES cycles (paper used 25 layers on 130 samples;
     we use far fewer -- see the deliberate-deviation note below).
  3. Measurement: Pauli-Z expectation on exactly 5 qubits (one per class,
     matching the paper's Eq. 3), fed through softmax -> categorical
     cross-entropy (paper's Eq. 4). No classical Dense head is added --
     this keeps the classifier a "true" VQC rather than a quantum-feature-
     extractor-plus-classical-head hybrid.
  4. Optimizer: the paper used plain Gradient Descent. We default to
     Adam instead (qml.AdamOptimizer) since it converges faster/more
     reliably with our much smaller epoch budget -- swap the line marked
     below if you want strict fidelity to the cited paper.
  5. Class-balanced loss: an earlier version of this script used a plain
     unweighted mean cross-entropy. On the real GSE7869 data that caused
     the circuit to collapse onto the majority classes (predicted
     Large_Cyst 0/3 times and Normal_Control 1/3 times across all 21
     folds -- exactly the two smallest classes). The loss now applies
     the same per-fold class-balancing weights phase3 uses for
     RF/XGBoost/MLP, so this is now a fair apples-to-apples comparison
     on that front. Per-fold training accuracy is also logged now
     (train_acc column) so you can tell overfitting (train_acc~1.0,
     LOOCV accuracy low) apart from under-convergence (train_acc low
     too) apart from a fixed collapse (train_acc high but always
     predicting 1-2 classes) if results still look off.

IMPORTANT, READ BEFORE RUNNING -- methodological caveats we are flagging
rather than hiding:
  - Parameter count vs. sample size: N_CYCLES=3 already gives 2*3*7*2=84
    trainable circuit parameters, fit on only 20 LOOCV training samples
    per fold. This is a real overfitting risk. The cited paper itself
    showed a train/val accuracy gap (0.88 vs 0.79) with 130 samples and
    25 layers -- expect a *larger* gap here with 21 samples. Watch the
    printed per-fold training cost: if it is trending to ~0 while LOOCV
    accuracy stays low, that IS overfitting, not a bug. Try N_CYCLES=2
    first if that happens.
  - This is noiseless, analytic simulation (default.qubit, shots=None) --
    it says nothing about how the circuit would behave on real NISQ
    hardware with shot noise and decoherence. Flag this in your paper.
  - Same patient-leakage caveat as phase3: several of the 21 samples
    likely share a patient (5 kidneys for 18 cyst/MCT samples per GEO's
    own summary). This script reuses the identical LOOCV folds as
    phase3, so the caveat and its consequences transfer unchanged.
  - mRMR feature selection is re-run from scratch in this script (not
    reused from phase3) so this file stays standalone. That means the
    full study reruns mRMR twice across the two scripts. If that
    duplicated cost is ever a problem, cache the per-fold selected gene
    lists to disk once and load them in both scripts.

Install once:
    pip install pennylane mrmr-selection scikit-learn pandas numpy
(No TensorFlow dependency for this script -- see note in the project
write-up: qml.qnn.KerasLayer no longer exists in current PennyLane, so
training uses PennyLane's own native optimizer / autograd interface
instead, which is also a closer match to the paper's own "Gradient
Descent Optimizer" description.)
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from mrmr import mrmr_classif

# ============================================================
# Config
# ============================================================
INPUT_CSV = "gene_expression_labeled.csv"   # same file phase3 uses
N_FEATURES = 75          # same mRMR count as phase3, for a fair comparison
PREFILTER = 3000         # same cheap variance pre-filter as phase3
N_QUBITS = int(np.ceil(np.log2(N_FEATURES)))     # 7, since 2^7=128 >= 75
N_CYCLES = 3             # 1 cycle = HEA1 + HEA2. Paper used 25 layers on
                         # 130 samples; kept far shallower here on purpose
                         # (see overfitting note above) -- raise cautiously.
VQC_EPOCHS = 60
LEARNING_RATE = 0.1
RANDOM_STATE = 42
OPTIMIZER = "adam"       # "adam" (faster/default) or "gd" (paper-faithful
                         # plain qml.GradientDescentOptimizer)

np.random.seed(RANDOM_STATE)

# ============================================================
# Load (identical pattern to phase3)
# ============================================================
df = pd.read_csv(INPUT_CSV, index_col=0)
X = df.drop(columns=["label"]).reset_index(drop=True)
y_raw = df["label"].reset_index(drop=True)

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_
N_READOUT = len(class_names)   # 5 for our dataset

assert N_READOUT <= N_QUBITS, (
    f"Need at least as many qubits ({N_QUBITS}) as classes ({N_READOUT}) "
    "to give every class its own readout qubit."
)
assert N_FEATURES <= 2 ** N_QUBITS, "N_FEATURES must fit in 2**N_QUBITS amplitudes."

print("Classes:", list(class_names))
print(f"N_QUBITS={N_QUBITS} (amplitude-encoding {N_FEATURES} genes, padded to {2**N_QUBITS})")
print(f"N_READOUT={N_READOUT} qubits measured (1 per class)")
print(f"N_CYCLES={N_CYCLES} -> {2 * N_CYCLES * N_QUBITS * 2} trainable circuit parameters")
print(f"Running Leave-One-Out over {len(X)} samples x {X.shape[1]} genes...\n")

# ============================================================
# Quantum circuit: feature mapping + two HEAs + 5-qubit Z readout
# ============================================================
dev = qml.device("default.qubit", wires=N_QUBITS)


def hea1(params, wires):
    """HEA #1 from the paper: Hadamard + RX,RY per qubit, then CNOT ring
    + Toffoli ring entanglers."""
    n = len(wires)
    for w in wires:
        qml.Hadamard(wires=w)
    for i, w in enumerate(wires):
        qml.RX(params[i, 0], wires=w)
        qml.RY(params[i, 1], wires=w)
    for i in range(n):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])
    for i in range(n):
        qml.Toffoli(wires=[wires[i], wires[(i + 1) % n], wires[(i + 2) % n]])


def hea2(params, wires):
    """HEA #2 from the paper: Hadamard + RY,RZ per qubit, then the same
    CNOT ring + Toffoli ring entanglers."""
    n = len(wires)
    for w in wires:
        qml.Hadamard(wires=w)
    for i, w in enumerate(wires):
        qml.RY(params[i, 0], wires=w)
        qml.RZ(params[i, 1], wires=w)
    for i in range(n):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])
    for i in range(n):
        qml.Toffoli(wires=[wires[i], wires[(i + 1) % n], wires[(i + 2) % n]])


@qml.qnode(dev, diff_method="backprop")
def circuit(x, weights1, weights2):
    """x: (batch, N_FEATURES) or (N_FEATURES,). Returns a list of
    N_READOUT PauliZ expectation values (batched -> each entry has shape
    (batch,))."""
    qml.AmplitudeEmbedding(x, wires=range(N_QUBITS), normalize=True, pad_with=0.0)
    for l in range(N_CYCLES):
        hea1(weights1[l], wires=range(N_QUBITS))
        hea2(weights2[l], wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_READOUT)]


def forward_probs(X_batch, weights1, weights2):
    """Softmax over the N_READOUT qubit Z-expectations (paper's Eq. 3).
    Returns shape (batch, N_READOUT)."""
    raw = circuit(X_batch, weights1, weights2)     # (N_READOUT, batch)
    z = pnp.stack(raw).T                           # (batch, N_READOUT)
    z = z - pnp.max(z, axis=1, keepdims=True)       # numerical stability
    e = pnp.exp(z)
    return e / pnp.sum(e, axis=1, keepdims=True)


def batch_cost(weights1, weights2, X_batch, y_int, sample_weight):
    """Class-balanced categorical cross-entropy over a batch (paper's Eq. 4,
    with the same class-balancing scheme phase3 uses for RF/XGB/MLP -- see
    weight_by_class below). Without this, classes with more training samples
    dominate the average loss and the circuit learns to just favor them,
    which is exactly the majority-class collapse seen in initial testing on
    the real 5/5/3/5/3-imbalanced dataset."""
    probs = forward_probs(X_batch, weights1, weights2)
    n = X_batch.shape[0]
    correct_probs = pnp.stack([probs[i, y_int[i]] for i in range(n)])
    losses = -pnp.log(correct_probs + 1e-10)
    weighted = losses * sample_weight
    return pnp.sum(weighted) / pnp.sum(sample_weight)


def make_optimizer():
    if OPTIMIZER == "adam":
        return qml.AdamOptimizer(stepsize=LEARNING_RATE)
    elif OPTIMIZER == "gd":
        return qml.GradientDescentOptimizer(stepsize=LEARNING_RATE)
    raise ValueError("OPTIMIZER must be 'adam' or 'gd'")


def init_weights():
    w1 = pnp.array(np.random.uniform(0, 2 * np.pi, (N_CYCLES, N_QUBITS, 2)), requires_grad=True)
    w2 = pnp.array(np.random.uniform(0, 2 * np.pi, (N_CYCLES, N_QUBITS, 2)), requires_grad=True)
    return w1, w2


# ============================================================
# Nested Leave-One-Out loop (mirrors phase3 exactly, so fold i in
# phase3 and fold i here hold out the *same* sample)
# ============================================================
loo = LeaveOneOut()
vqc_results = []
fold_log = []
proba_log = []   

t_start = time.perf_counter()
for fold, (train_idx, test_idx) in enumerate(loo.split(X), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ---- feature selection: fit on this fold's training data ONLY (identical to phase3) ----
    top_var_genes = X_train.var().sort_values(ascending=False).index[:PREFILTER]
    X_train_pf, X_test_pf = X_train[top_var_genes], X_test[top_var_genes]

    y_train_series = pd.Series(y_train, index=X_train_pf.index)
    selected = mrmr_classif(X=X_train_pf, y=y_train_series, K=N_FEATURES, show_progress=False)
    X_train_sel, X_test_sel = X_train_pf[selected].values, X_test_pf[selected].values

    # ---- normalization: StandardScaler then MinMaxScaler([0,1]), fit on TRAIN only ----
    # (this exact two-step normalization mirrors arXiv:2505.02033's preprocessing,
    # important for amplitude embedding since raw expression scale would otherwise
    # let a few high-magnitude genes dominate the resulting quantum state)
    std_scaler = StandardScaler().fit(X_train_sel)
    X_train_std = std_scaler.transform(X_train_sel)
    X_test_std = std_scaler.transform(X_test_sel)

    mm_scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_train_std)
    X_train_scaled = mm_scaler.transform(X_train_std)
    X_test_scaled = mm_scaler.transform(X_test_std)

    # ---- class weights: computed on this fold's training labels ONLY (identical scheme to phase3) ----
    classes, counts = np.unique(y_train, return_counts=True)
    weight_by_class = {c: len(y_train) / (len(classes) * n) for c, n in zip(classes, counts)}
    sample_weight = np.array([weight_by_class[label] for label in y_train])

    # ---- fresh VQC each fold (avoid leaking learned params across folds) ----
    weights1, weights2 = init_weights()
    opt = make_optimizer()

    for epoch in range(VQC_EPOCHS):
        (weights1, weights2), train_cost = opt.step_and_cost(
            lambda w1, w2: batch_cost(w1, w2, X_train_scaled, y_train, sample_weight), weights1, weights2
        )

    train_probs = np.array(forward_probs(X_train_scaled, weights1, weights2))
    train_acc = (np.argmax(train_probs, axis=1) == y_train).mean()

    test_probs = np.array(forward_probs(X_test_scaled, weights1, weights2))
    vqc_pred = int(np.argmax(test_probs[0]))
    vqc_results.append((y_test[0], vqc_pred))

    true_label = class_names[y_test[0]]
    print(f"Fold {fold:2d}/{len(X)}  true={true_label:15s}  "
          f"VQC={class_names[vqc_pred]:15s}  final_train_cost={float(train_cost):.4f}  "
          f"train_acc={train_acc:.3f}")

    fold_log.append({
        "fold": fold, "true_label": true_label,
        "VQC_pred": class_names[vqc_pred],
        "final_train_cost": float(train_cost),
        "train_acc": float(train_acc),
    })

# NEW: stash VQC probabilities for the combined ROC figure
    proba_row = {"fold": fold, "true_label": true_label}
    for cls_idx, cls_name in enumerate(class_names):
        proba_row[f"VQC_{cls_name}"] = float(test_probs[0][cls_idx])
    proba_log.append(proba_row)

t_end = time.perf_counter()
print(f"\nTotal VQC training+eval time for {len(X)} folds: {(t_end - t_start):.1f}s")

# ============================================================
# Report + save (same style as phase3)
# ============================================================
print("\n" + "=" * 60)
y_true = [p[0] for p in vqc_results]
y_pred = [p[1] for p in vqc_results]
print("\n=== Deep VQC (Leave-One-Out, {} folds) ===".format(len(X)))
print(f"Overall accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

pd.DataFrame(fold_log).to_csv("vqc_loocv_fold_results.csv", index=False)
pd.DataFrame(proba_log).to_csv("vqc_fold_probabilities.csv", index=False)
print("Saved: vqc_fold_probabilities.csv (per-class VQC probabilities)")
print("Per-fold predictions saved to vqc_loocv_fold_results.csv")

# ============================================================
# Optional: side-by-side comparison with phase3's classical results,
# if that CSV is sitting in the same folder (run phase3 first if not)
# ============================================================
try:
    classical = pd.read_csv("loocv_fold_results.csv")
    merged = classical.merge(pd.DataFrame(fold_log)[["fold", "VQC_pred"]], on="fold")
    print("\n=== Combined comparison (classical models + VQC) ===")
    for col in ["RF_pred", "XGB_pred", "MLP_pred", "VQC_pred"]:
        acc = (merged[col] == merged["true_label"]).mean()
        print(f"{col:10s} LOOCV accuracy: {acc:.4f}")
    merged.to_csv("combined_loocv_comparison.csv", index=False)
    print("Combined table saved to combined_loocv_comparison.csv")
except FileNotFoundError:
    print("\n(loocv_fold_results.csv not found in this folder -- run "
          "phase3_mrmr_loocv_models.py first if you want the combined "
          "classical-vs-quantum comparison table.)")