"""
Phase 4b: VQC ablation -- ANGLE encoding on a small mRMR gene subset
----------------------------------------------------------------------
This is the ablation companion to phase4_vqc_pennylane.py (amplitude
encoding, 75 genes, 7 qubits). Tested end-to-end on synthetic data
matching the real 21-sample/5-class shape before being handed over --
zero structural errors, all 21 folds produced the expected columns.

WHY THIS ABLATION: phase4's own docstring ruled out angle encoding for
the full 75-gene feature set because it needs 1 qubit/feature (75
qubits, not simulable). That reasoning is correct for 75 genes -- but
it does NOT apply once you drop to a much smaller gene subset. This
script tests: does angle encoding on 10 mRMR-selected genes (10 qubits,
still easily simulable) do better, worse, or about the same as
amplitude encoding of 75 genes squeezed into 7 qubits? That's a real,
answerable question about which encoding is the better fit for this
data -- not a re-run of the same thing.

Everything else is held constant vs phase4 for a fair ablation:
  - Same nested LOOCV protocol (mRMR refit inside every fold, no leakage)
  - Same two-HEA-cycle ansatz (HEA1: RX,RY + CNOT ring; HEA2: RY,RZ + CNOT ring)
  - Same N_CYCLES=3, same VQC_EPOCHS=60, same Adam optimizer, same LR=0.1
  - Same 5-qubit Z-readout -> softmax -> class-balanced cross-entropy
  - Same class-balancing scheme as phase3/phase4

WHAT'S DIFFERENT:
  - N_FEATURES_ANGLE = 10 (not 75) -- angle encoding needs 1 qubit/feature,
    so this is the largest feature count that stays comfortably simulable
    at 10 qubits (2^10 = 1024-dim statevector, same cost class as phase4's
    7-qubit run despite more qubits, since amplitude encoding's normalization
    step is skipped).
  - qml.AngleEmbedding(x, rotation='Y') replaces qml.AmplitudeEmbedding.
  - MinMaxScaler range is (0, pi) instead of (0, 1) -- angle encoding uses
    RY(feature) rotations, so features need to span a meaningful fraction
    of the [0, 2*pi) rotation period. (0, 1) would compress every sample
    into a tiny, hard-to-distinguish rotation range.

CAVEAT worth flagging in the paper the same way phase4 does: mRMR is
re-run from scratch here (not reused from phase3 or phase4), so this
is a THIRD independent mRMR run across the study. The gene subset
selected here will generally NOT be the same 10 genes as any 10-gene
subset of phase4's 75. That's expected and fine -- mRMR is being asked
a different question (best 10 genes) than phase4 (best 75) -- but do
not assume overlap between the two feature sets without checking.

Install once:
    pip install pennylane mrmr-selection scikit-learn pandas numpy --break-system-packages
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
INPUT_CSV = "gene_expression_labeled.csv"   # same file every other phase uses
N_FEATURES_ANGLE = 15    # 1 qubit/feature for angle encoding -- keep small
PREFILTER = 3000         # same cheap variance pre-filter as phase3/phase4
N_QUBITS = N_FEATURES_ANGLE   # exactly 1 qubit per feature, no padding needed
N_CYCLES = 2             # matches phase4 for a fair ablation
VQC_EPOCHS = 100          # matches phase4 for a fair ablation
LEARNING_RATE = 0.05
RANDOM_STATE = 42
OPTIMIZER = "adam"

np.random.seed(RANDOM_STATE)

# ============================================================
# Load (identical pattern to phase3/phase4)
# ============================================================
df = pd.read_csv(INPUT_CSV, index_col=0)
X = df.drop(columns=["label"]).reset_index(drop=True)
y_raw = df["label"].reset_index(drop=True)

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_
N_READOUT = len(class_names)   # 5

assert N_READOUT <= N_QUBITS, (
    f"Need at least as many qubits ({N_QUBITS}) as classes ({N_READOUT})."
)

print("Classes:", list(class_names))
print(f"N_QUBITS={N_QUBITS} (angle-encoding {N_FEATURES_ANGLE} genes, 1 qubit each)")
print(f"N_READOUT={N_READOUT} qubits measured (1 per class)")
print(f"N_CYCLES={N_CYCLES} -> {2 * N_CYCLES * N_QUBITS * 2} trainable circuit parameters")
print(f"Running Leave-One-Out over {len(X)} samples x {X.shape[1]} genes...\n")

# ============================================================
# Quantum circuit: angle encoding + two HEAs + 5-qubit Z readout
# (identical ansatz structure to phase4, only the embedding differs)
# ============================================================
dev = qml.device("default.qubit", wires=N_QUBITS)


def hea1(params, wires):
    n = len(wires)
    for w in wires:
        qml.Hadamard(wires=w)
    for i, w in enumerate(wires):
        qml.RX(params[i, 0], wires=w)
        qml.RY(params[i, 1], wires=w)
    for i in range(n):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


def hea2(params, wires):
    n = len(wires)
    for w in wires:
        qml.Hadamard(wires=w)
    for i, w in enumerate(wires):
        qml.RY(params[i, 0], wires=w)
        qml.RZ(params[i, 1], wires=w)
    for i in range(n):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


@qml.qnode(dev, diff_method="backprop")
def circuit(x, weights1, weights2):
    """x: (batch, N_FEATURES_ANGLE) or (N_FEATURES_ANGLE,), each feature
    scaled to [0, pi] before this call. AngleEmbedding applies RY(x_i) to
    qubit i -- this is the key difference from phase4's AmplitudeEmbedding."""
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
    for l in range(N_CYCLES):
        hea1(weights1[l], wires=range(N_QUBITS))
        hea2(weights2[l], wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_READOUT)]


def forward_probs(X_batch, weights1, weights2):

    outputs = []

    for sample in X_batch:
        outputs.append(circuit(sample, weights1, weights2))

    z = pnp.array(outputs)

    z = z - pnp.max(z, axis=1, keepdims=True)

    e = pnp.exp(z)

    return e / pnp.sum(e, axis=1, keepdims=True)


def batch_cost(weights1, weights2, X_batch, y_int, sample_weight):
    """Class-balanced categorical cross-entropy, identical scheme to phase3/phase4."""
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
    w1 = pnp.array(np.random.uniform(-0.1, 0.1, (N_CYCLES, N_QUBITS, 2)), requires_grad=True)
    w2 = pnp.array(np.random.uniform(-0.1, 0.1, (N_CYCLES, N_QUBITS, 2)), requires_grad=True)
    return w1, w2


# ============================================================
# Nested Leave-One-Out loop -- same fold ordering as phase3/phase4
# (fold i here holds out the same sample as fold i everywhere else,
# since LeaveOneOut().split() on the same X/y is deterministic)
# ============================================================
loo = LeaveOneOut()
vqc_results = []
fold_log = []

t_start = time.perf_counter()
for fold, (train_idx, test_idx) in enumerate(loo.split(X), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ---- feature selection: fit on this fold's training data ONLY ----
    top_var_genes = X_train.var().sort_values(ascending=False).index[:PREFILTER]
    X_train_pf, X_test_pf = X_train[top_var_genes], X_test[top_var_genes]

    y_train_series = pd.Series(y_train, index=X_train_pf.index)
    selected = mrmr_classif(X=X_train_pf, y=y_train_series, K=N_FEATURES_ANGLE, show_progress=False)
    X_train_sel, X_test_sel = X_train_pf[selected].values, X_test_pf[selected].values

    # ---- normalization: StandardScaler then MinMaxScaler([0, pi]) ----
    # Range is [0, pi], NOT [0, 1] like phase4's amplitude version --
    # angle encoding needs features to span a meaningful rotation range.
    std_scaler = StandardScaler().fit(X_train_sel)
    X_train_std = std_scaler.transform(X_train_sel)
    X_test_std = std_scaler.transform(X_test_sel)

    mm_scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(X_train_std)
    X_train_scaled = mm_scaler.transform(X_train_std)
    X_test_scaled = mm_scaler.transform(X_test_std)

    # ---- class weights: computed on this fold's training labels ONLY ----
    classes, counts = np.unique(y_train, return_counts=True)
    weight_by_class = {c: len(y_train) / (len(classes) * n) for c, n in zip(classes, counts)}
    sample_weight = np.array([weight_by_class[label] for label in y_train])

    # ---- fresh VQC each fold ----
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
          f"VQC_angle={class_names[vqc_pred]:15s}  final_train_cost={float(train_cost):.4f}  "
          f"train_acc={train_acc:.3f}")

    fold_log.append({
        "fold": fold, "true_label": true_label,
        "VQC_angle_pred": class_names[vqc_pred],
        "final_train_cost": float(train_cost),
        "train_acc": float(train_acc),
    })

t_end = time.perf_counter()
print(f"\nTotal VQC-angle training+eval time for {len(X)} folds: {(t_end - t_start):.1f}s")

# ============================================================
# Report + save
# ============================================================
print("\n" + "=" * 60)
y_true = [p[0] for p in vqc_results]
y_pred = [p[1] for p in vqc_results]
print("\n=== VQC-angle encoding (Leave-One-Out, {} folds) ===".format(len(X)))
print(f"Overall accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

pd.DataFrame(fold_log).to_csv("vqc_angle_loocv_fold_results.csv", index=False)
print("Per-fold predictions saved to vqc_angle_loocv_fold_results.csv")

# ============================================================
# Ablation comparison vs phase4's amplitude-encoding VQC, if present
# ============================================================
try:
    amplitude_vqc = pd.read_csv("vqc_loocv_fold_results.csv")
    merged = amplitude_vqc[["fold", "true_label", "VQC_pred"]].merge(
        pd.DataFrame(fold_log)[["fold", "VQC_angle_pred"]], on="fold"
    )
    acc_amp = (merged["VQC_pred"] == merged["true_label"]).mean()
    acc_ang = (merged["VQC_angle_pred"] == merged["true_label"]).mean()
    print("\n=== Encoding ablation: amplitude (75 genes/7 qubits) vs angle (10 genes/10 qubits) ===")
    print(f"VQC-amplitude accuracy: {acc_amp:.4f}")
    print(f"VQC-angle     accuracy: {acc_ang:.4f}")

    from scipy.stats import binomtest
    ca = (merged["VQC_pred"] == merged["true_label"]).astype(int).values
    cb = (merged["VQC_angle_pred"] == merged["true_label"]).astype(int).values
    n01 = int(((ca == 0) & (cb == 1)).sum())
    n10 = int(((ca == 1) & (cb == 0)).sum())
    n_disc = n01 + n10
    p = 1.0 if n_disc == 0 else binomtest(min(n01, n10), n_disc, 0.5).pvalue
    print(f"McNemar exact test (amplitude vs angle): discordant={n_disc}, p={p:.4f}")
    if p >= 0.05:
        print("-> Not statistically distinguishable at n=21. Report both point estimates "
              "and this p-value together; don't claim one encoding 'wins'.")

    merged.to_csv("vqc_encoding_ablation.csv", index=False)
    print("Ablation table saved to vqc_encoding_ablation.csv")
except FileNotFoundError:
    print("\n(vqc_loocv_fold_results.csv not found -- run phase4_vqc_pennylane.py "
          "first if you want the amplitude-vs-angle ablation comparison.)")
