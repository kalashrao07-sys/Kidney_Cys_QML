"""
Merges phase5_fold_probabilities.csv (RF/XGB/SVM/MLP), 
phase6_qsvm_fold_probabilities.csv (QSVM), and vqc_fold_probabilities.csv (VQC)
into all_models_fold_probabilities.csv, which Generate_paper_figures.py's
figure_roc_curves() reads.

Run once, after re-running phase4/5/6 with their probability-saving patches applied:
    python build_roc_probabilities.py
"""
import pandas as pd

p5 = pd.read_csv("phase5_fold_probabilities.csv")
p6 = pd.read_csv("phase6_qsvm_fold_probabilities.csv")
p4 = pd.read_csv("vqc_fold_probabilities.csv")

merged = p5.copy()
for other, name in [(p6, "phase6"), (p4, "phase4")]:
    check = merged.merge(other[["fold", "true_label"]], on="fold", suffixes=("", "_chk"))
    mismatches = check[check["true_label"] != check["true_label_chk"]]
    if len(mismatches) > 0:
        raise ValueError(f"Fold/true_label mismatch merging {name} -- "
                          f"these scripts are NOT using the same LOOCV fold ordering:\n{mismatches}")
    other_cols = [c for c in other.columns if c not in ("fold", "true_label")]
    merged = merged.merge(other[["fold"] + other_cols], on="fold")

merged.to_csv("all_models_fold_probabilities.csv", index=False)
print(f"Saved all_models_fold_probabilities.csv ({len(merged)} folds, "
      f"{len(merged.columns)} columns)")