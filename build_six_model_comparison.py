"""
Final six-model comparison builder: RF, XGBoost, MLP, SVM-RBF, QSVM, VQC
--------------------------------------------------------------------------
Run this once you have ALL of the following fold-level result files in the
same folder (all must share the same 21-fold ordering as gene_expression_labeled.csv,
since LOOCV fold i must mean "sample i held out" identically across scripts):

  - loocv_fold_results.csv           (from phase3: RF_pred, XGB_pred, MLP_pred)
  - <svm_fold_results>.csv           (from phase5: needs an SVM_pred column -- EDIT FILENAME BELOW)
  - <qsvm_fold_results>.csv          (from phase6: needs a QSVM_pred column -- EDIT FILENAME BELOW)
  - vqc_loocv_fold_results.csv       (from phase4: VQC_pred)

EDIT the three filename variables below to match what phase5/phase6 actually
wrote to disk, then run:
    python build_six_model_comparison.py

Install once:
    pip install pandas scikit-learn scipy --break-system-packages
"""

import pandas as pd
from itertools import combinations
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from scipy.stats import binomtest

# ============================================================
# EDIT THESE THREE FILENAMES to match your actual phase5/phase6 outputs
# ============================================================
PHASE3_CSV = "loocv_fold_results.csv"          # RF_pred, XGB_pred, MLP_pred
PHASE5_SVM_CSV = "phase5_loocv_fold_results.csv"  # must contain an SVM_pred column
PHASE6_QSVM_CSV = "phase6_qsvm_fold_results.csv"  # must contain a QSVM_pred column
PHASE4_VQC_CSV = "vqc_loocv_fold_results.csv"  # VQC_pred

MODEL_COLUMNS = ["RF_pred", "XGB_pred", "MLP_pred", "SVM_pred", "QSVM_pred", "VQC_pred"]
LABELS = ["Small_Cyst", "Medium_Cyst", "Large_Cyst", "MCT", "Normal_Control"]

# ============================================================
# Load and merge on fold number (sanity-checks true_label agreement)
# ============================================================
p3 = pd.read_csv(PHASE3_CSV)
p5 = pd.read_csv(PHASE5_SVM_CSV)
p6 = pd.read_csv(PHASE6_QSVM_CSV)
p4 = pd.read_csv(PHASE4_VQC_CSV)

merged = p3[["fold", "true_label", "RF_pred", "XGB_pred", "MLP_pred"]].copy()
for other, col in [(p5, "SVM_pred"), (p6, "QSVM_pred"), (p4, "VQC_pred")]:
    sub = other[["fold", "true_label", col]]
    check = merged.merge(sub[["fold", "true_label"]], on="fold", suffixes=("", "_chk"))
    mismatches = check[check["true_label"] != check["true_label_chk"]]
    if len(mismatches) > 0:
        raise ValueError(
            f"Fold/true_label mismatch when merging {col} -- these scripts are NOT "
            f"using the same LOOCV fold ordering. Mismatched folds:\n{mismatches}"
        )
    merged = merged.merge(sub[["fold", col]], on="fold")

merged.to_csv("final_six_model_comparison.csv", index=False)
print(f"Merged {len(merged)} folds. Saved to final_six_model_comparison.csv\n")

# ============================================================
# Per-model accuracy / F1 / classification report
# ============================================================
print("=" * 70)
print("PER-MODEL PERFORMANCE (Leave-One-Out, 21 folds)")
print("=" * 70)
results_summary = []
for col in MODEL_COLUMNS:
    acc = accuracy_score(merged["true_label"], merged[col])
    f1 = f1_score(merged["true_label"], merged[col], average="weighted")
    results_summary.append({"model": col.replace("_pred", ""), "accuracy": acc, "weighted_f1": f1})
    print(f"\n--- {col} ---")
    print(f"Accuracy: {acc:.4f}   Weighted F1: {f1:.4f}")
    print(classification_report(merged["true_label"], merged[col], labels=LABELS, zero_division=0))

summary_df = pd.DataFrame(results_summary).sort_values("accuracy", ascending=False)
print("\n" + "=" * 70)
print("SUMMARY TABLE (sorted by accuracy)")
print("=" * 70)
print(summary_df.to_string(index=False))
summary_df.to_csv("final_six_model_summary.csv", index=False)

# ============================================================
# Confusion matrices (for the paper's error-profile discussion)
# ============================================================
print("\n" + "=" * 70)
print("CONFUSION MATRICES (rows=true, cols=pred)")
print("=" * 70)
for col in MODEL_COLUMNS:
    print(f"\n--- {col} --- labels order: {LABELS}")
    cm = confusion_matrix(merged["true_label"], merged[col], labels=LABELS)
    print(cm)

# ============================================================
# Pairwise exact McNemar tests -- essential given n=21
# ============================================================
print("\n" + "=" * 70)
print("PAIRWISE EXACT MCNEMAR TESTS (is the accuracy difference real at n=21?)")
print("=" * 70)
correct = {
    col: (merged[col] == merged["true_label"]).astype(int).values
    for col in MODEL_COLUMNS
}
mcnemar_rows = []
for a, b in combinations(MODEL_COLUMNS, 2):
    ca, cb = correct[a], correct[b]
    n01 = int(((ca == 0) & (cb == 1)).sum())
    n10 = int(((ca == 1) & (cb == 0)).sum())
    n_disc = n01 + n10
    p = 1.0 if n_disc == 0 else binomtest(min(n01, n10), n_disc, 0.5).pvalue
    mcnemar_rows.append(
        {"model_a": a, "model_b": b, "discordant_folds": n_disc, "p_value": p}
    )
    flag = "  <-- significant (p<0.05)" if p < 0.05 else ""
    print(f"{a:10s} vs {b:10s}: discordant={n_disc:2d}  p={p:.4f}{flag}")

pd.DataFrame(mcnemar_rows).to_csv("final_six_model_mcnemar.csv", index=False)

print(
    "\nInterpretation guide: with only 21 folds, McNemar's test has low power. "
    "A non-significant p-value does NOT mean two models are equivalent -- it means "
    "you can't reject that possibility with this sample size. Report both the point "
    "estimates AND the p-values; don't claim a model is 'better' on accuracy alone "
    "if the difference isn't significant."
)
