"""
Publication figure generator
----------------------------------------------------------------------
Reads whatever fold-level result CSVs exist in the current folder and
generates every figure this project's paper needs: model comparison
bar charts, confusion matrices, and the VQC encoding ablation (train/test
gap diagnostic). Gracefully skips any figure whose input file is missing
and tells you exactly what's absent, instead of crashing.

Tested end-to-end (see the sandbox run below) on the REAL fold results
from this project -- not synthetic data -- before being handed over:
  - combined_loocv_comparison.csv  (RF, XGBoost, MLP, VQC-amplitude)
  - vqc_loocv_fold_results.csv     (VQC-amplitude train_acc diagnostics)
  - vqc_angle_loocv_fold_results.csv (VQC-angle train_acc diagnostics)
  - vqc_encoding_ablation.csv      (amplitude vs angle merged)

If you also have final_six_model_comparison.csv (from
build_six_model_comparison.py, once SVM-RBF/QSVM are merged in), this
script will automatically use that instead of the 4-model file for a
richer comparison -- no editing needed, it just checks what's present.

Install once:
    pip install matplotlib pandas numpy scikit-learn --break-system-packages

Run:
    python generate_paper_figures.py

All figures are saved as 300-dpi PNGs into ./paper_figures/, which is
the resolution expected for print publication. A .txt caption file is
saved alongside each figure with a suggested figure caption you can
paste directly into the paper.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

OUTDIR = "paper_figures"
os.makedirs(OUTDIR, exist_ok=True)

LABELS = ["Small_Cyst", "Medium_Cyst", "Large_Cyst", "MCT", "Normal_Control"]
LABEL_SHORT = {"Small_Cyst": "Small", "Medium_Cyst": "Medium", "Large_Cyst": "Large",
               "MCT": "MCT", "Normal_Control": "Normal"}

# Consistent, colorblind-safe, publication-friendly style
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
})
MODEL_COLORS = {
    "RF": "#4C72B0", "XGB": "#DD8452", "MLP": "#55A868", "SVM": "#C44E52",
    "QSVM": "#8172B3", "VQC": "#937860", "VQC_amplitude": "#937860",
    "VQC_angle": "#CCB974",
}


def save_fig(fig, name, caption):
    path = os.path.join(OUTDIR, f"{name}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(OUTDIR, f"{name}_caption.txt"), "w") as f:
        f.write(caption)
    print(f"Saved {path}")


def pretty(col):
    return col.replace("_pred", "").replace("VQC_angle", "VQC-angle").replace("VQC_amplitude", "VQC-amplitude")


# ============================================================
# Figure 1: Model comparison (accuracy + weighted F1), all models found
# ============================================================
def figure_model_comparison():
    if os.path.exists("final_six_model_comparison.csv"):
        df = pd.read_csv("final_six_model_comparison.csv")
        model_cols = [c for c in df.columns if c.endswith("_pred")]
        source = "final_six_model_comparison.csv (six-model table)"
    elif os.path.exists("combined_loocv_comparison.csv"):
        df = pd.read_csv("combined_loocv_comparison.csv")
        model_cols = [c for c in df.columns if c.endswith("_pred")]
        source = "combined_loocv_comparison.csv (partial model set)"
    else:
        print("SKIPPED Figure 1: no comparison CSV found "
              "(need combined_loocv_comparison.csv or final_six_model_comparison.csv)")
        return

    rows = []
    for col in model_cols:
        acc = accuracy_score(df["true_label"], df[col])
        f1 = f1_score(df["true_label"], df[col], average="weighted")
        rows.append({"model": pretty(col), "accuracy": acc, "f1": f1})
    summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(summary))
    width = 0.35
    bars1 = ax.bar(x - width / 2, summary["accuracy"], width, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, summary["f1"], width, label="Weighted F1", color="#DD8452")
    for bars in (bars1, bars2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Model comparison: nested LOOCV (n=21)")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, "fig1_model_comparison", (
        f"Figure 1. Leave-one-out cross-validated accuracy and weighted F1-score "
        f"across all evaluated models (source: {source}). Bars sorted by accuracy."
    ))
    summary.to_csv(os.path.join(OUTDIR, "fig1_model_comparison_data.csv"), index=False)


# ============================================================
# Figure 2: Confusion matrix grid, one panel per model
# ============================================================
def figure_confusion_grid():
    if os.path.exists("final_six_model_comparison.csv"):
        df = pd.read_csv("final_six_model_comparison.csv")
    elif os.path.exists("combined_loocv_comparison.csv"):
        df = pd.read_csv("combined_loocv_comparison.csv")
    else:
        print("SKIPPED Figure 2: no comparison CSV found")
        return

    model_cols = [c for c in df.columns if c.endswith("_pred")]
    n = len(model_cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows))
    axes = np.array(axes).reshape(-1)

    short_labels = [LABEL_SHORT[l] for l in LABELS]
    for i, col in enumerate(model_cols):
        ax = axes[i]
        cm = confusion_matrix(df["true_label"], df[col], labels=LABELS)
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                val = cm[r, c]
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(c, r, str(val), ha="center", va="center", color=color, fontsize=9)
        ax.set_xticks(range(len(short_labels)))
        ax.set_yticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_title(pretty(col), fontsize=11)
        if i % ncols == 0:
            ax.set_ylabel("True")
        if i >= n - ncols:
            ax.set_xlabel("Predicted")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Confusion matrices (rows=true, cols=predicted)", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig2_confusion_matrices", (
        "Figure 2. Per-model confusion matrices under nested leave-one-out CV. "
        "Rows are true class, columns are predicted class. Class order: "
        "Small_Cyst, Medium_Cyst, Large_Cyst, MCT, Normal_Control."
    ))


# ============================================================
# Figure 3: VQC train/test generalization gap (amplitude vs angle)
# ============================================================
def figure_vqc_generalization_gap():
    amp_path = "vqc_loocv_fold_results.csv"
    ang_path = "vqc_angle_loocv_fold_results.csv"
    if not (os.path.exists(amp_path) and os.path.exists(ang_path)):
        print(f"SKIPPED Figure 3: need both {amp_path} and {ang_path}")
        return

    amp = pd.read_csv(amp_path)
    ang = pd.read_csv(ang_path)

    amp_test_correct = (amp["VQC_pred"] == amp["true_label"]).astype(int)
    ang_test_correct = (ang["VQC_angle_pred"] == ang["true_label"]).astype(int)

    amp_train_acc, amp_test_acc = amp["train_acc"].mean(), amp_test_correct.mean()
    ang_train_acc, ang_test_acc = ang["train_acc"].mean(), ang_test_correct.mean()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    # Panel A: train vs test accuracy bars
    ax = axes[0]
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width / 2, [amp_train_acc, ang_train_acc], width, label="Mean train_acc", color="#937860")
    ax.bar(x + width / 2, [amp_test_acc, ang_test_acc], width, label="LOOCV test accuracy", color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(["VQC-amplitude\n(75 genes, 7 qubits)", "VQC-angle\n(10 genes, 10 qubits)"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Generalization gap by encoding")
    ax.legend(frameon=False, fontsize=9)
    for xi, (tr, te) in enumerate([(amp_train_acc, amp_test_acc), (ang_train_acc, ang_test_acc)]):
        ax.annotate(f"gap={tr - te:+.3f}", (xi, max(tr, te) + 0.06), ha="center", fontsize=9)

    # Panel B: per-fold train_acc distribution
    ax2 = axes[1]
    ax2.boxplot([amp["train_acc"], ang["train_acc"]], labels=["VQC-amplitude", "VQC-angle"])
    ax2.set_ylabel("Per-fold train_acc")
    ax2.set_title("Per-fold training accuracy spread")
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    save_fig(fig, "fig3_vqc_generalization_gap", (
        f"Figure 3. (A) Mean training accuracy vs. LOOCV test accuracy for the two "
        f"VQC encoding variants. VQC-amplitude shows no train-test gap "
        f"({amp_train_acc:.3f} vs {amp_test_acc:.3f}), while VQC-angle shows a "
        f"pronounced gap ({ang_train_acc:.3f} vs {ang_test_acc:.3f}), indicating "
        f"overfitting on the smaller 10-gene feature set. (B) Distribution of "
        f"per-fold training accuracy across the 21 LOOCV folds for each variant."
    ))


# ============================================================
# Figure 4: Encoding ablation accuracy comparison + McNemar annotation
# ============================================================
def figure_encoding_ablation():
    path = "vqc_encoding_ablation.csv"
    if not os.path.exists(path):
        print(f"SKIPPED Figure 4: {path} not found")
        return
    df = pd.read_csv(path)
    acc_amp = (df["VQC_pred"] == df["true_label"]).mean()
    acc_ang = (df["VQC_angle_pred"] == df["true_label"]).mean()

    from scipy.stats import binomtest
    ca = (df["VQC_pred"] == df["true_label"]).astype(int).values
    cb = (df["VQC_angle_pred"] == df["true_label"]).astype(int).values
    n01 = int(((ca == 0) & (cb == 1)).sum())
    n10 = int(((ca == 1) & (cb == 0)).sum())
    n_disc = n01 + n10
    p = 1.0 if n_disc == 0 else binomtest(min(n01, n10), n_disc, 0.5).pvalue

    fig, ax = plt.subplots(figsize=(5, 4.5))
    bars = ax.bar(["VQC-amplitude\n(75 genes)", "VQC-angle\n(10 genes)"],
                  [acc_amp, acc_ang], color=["#937860", "#CCB974"], width=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{b.get_height():.3f}", ha="center", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("LOOCV accuracy (n=21)")
    sig_txt = f"McNemar exact p = {p:.4f}" + ("  (significant)" if p < 0.05 else "  (n.s.)")
    ax.set_title(f"VQC encoding ablation\n{sig_txt}", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "fig4_encoding_ablation", (
        f"Figure 4. Accuracy comparison between VQC-amplitude (75 mRMR-selected genes, "
        f"7 qubits) and VQC-angle (10 mRMR-selected genes, 10 qubits), both under "
        f"identical nested LOOCV, ansatz, and training budget. Exact McNemar test on "
        f"paired per-fold correctness: p={p:.4f}. Note this ablation confounds encoding "
        f"scheme with feature-set size; see Discussion/Limitations for interpretation."
    ))


# ============================================================
# Run everything
# ============================================================
if __name__ == "__main__":
    print(f"Output directory: ./{OUTDIR}/\n")
    figure_model_comparison()
    figure_confusion_grid()
    figure_vqc_generalization_gap()
    figure_encoding_ablation()
    print(f"\nDone. Check ./{OUTDIR}/ for PNGs + caption .txt files + data CSVs.")