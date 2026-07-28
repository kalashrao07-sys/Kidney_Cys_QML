# Kidney Cyst Gene Expression Classification — Classical ML + Quantum ML (QML)

Extension of Patil, Akkasaligar & Pattar, *"Kidney Cyst Gene Expression
Analysis using Machine Learning"* (KLE Technological University) — the
original paper trained a single MLP (86.61% peak accuracy, no
cross-validation shown) on Affymetrix microarray data to classify
kidney tissue into five categories relevant to early Polycystic Kidney
Disease (PKD) detection.

This project reproduces that pipeline properly validated, benchmarks
Random Forest and XGBoost alongside the MLP, and adds a Variational
Quantum Classifier (VQC) to test whether a quantum model offers any
real advantage on this data.

**Status: research in progress.** Real results exist below, but two
things are explicitly unresolved (see *Known Issues*) — this is not a
finished, publication-ready result yet.

## Dataset

- **GSE7869** (NCBI GEO), platform GPL570 (Affymetrix Human Genome
  U133 Plus 2.0 Array), 21 samples, 54,675 probes.
- 5 classes, verified against GEO's own sample metadata: Small_Cyst
  (5), Medium_Cyst (5), Large_Cyst (3), MCT / minimally cystic tissue
  (5), Normal_Control (3).
- Not merged with any other dataset. GSE159566 could not be verified
  as a real/relevant accession. GSE35831 (a second human ADPKD
  dataset) was deliberately excluded from training — different
  microarray platform, and its samples are cultured cell lines rather
  than primary tissue, which would confound a pooled classifier.

## Repository structure

| File | Role |
|---|---|
| `preprocess.R` | Raw `.CEL.gz` → RMA normalization (background correction, quantile normalization, summarization) → labeled expression matrix (`gene_expression_labeled.csv`, 21 samples × 54,675 genes) |
| `phase3_mrmr_loocv_models.py` | Classical baseline: mRMR feature selection + Random Forest / XGBoost / MLP, evaluated with nested Leave-One-Out CV |
| `phase4_vqc_pennylane.py` | Variational Quantum Classifier (PennyLane), same nested LOOCV protocol, for a fair classical-vs-quantum comparison |
| `pqc_hybrid_encryption.py` | ML-KEM (NIST FIPS 203) + AES-256-GCM hybrid encryption for the dataset — optional/future-work only, **not** the project's quantum contribution |
| `loocv_fold_results.csv` | Per-fold predictions, classical models |
| `vqc_loocv_fold_results.csv` | Per-fold predictions, VQC |
| `combined_loocv_comparison.csv` | All four models side by side, same folds |

## Methodology

**Why Leave-One-Out CV:** the smallest classes (Large_Cyst,
Normal_Control) have only 3 samples each, which breaks stratified
k-fold. LOOCV guarantees every sample is tested exactly once.

**Why "nested":** mRMR feature selection (top ~75 genes, after a
cheap variance pre-filter to 3,000) and class-balance weighting are
both recomputed from scratch *inside every fold*, using only that
fold's 20 training samples — never on the full 21-sample set. Fitting
feature selection before splitting would let the held-out sample
quietly influence which genes get chosen, which is a common,
easy-to-miss source of leakage in small-sample gene-expression papers.

**Class imbalance:** handled via class weighting (`class_weight`,
`sample_weight`), not SMOTE — with classes as small as 3 real samples,
synthetic interpolation was judged unreliable.

## Results (Leave-One-Out, 21 folds, 5-class)

| Model | Accuracy |
|---|---|
| Random Forest | 17/21 — **81.0%** |
| MLP (paper's architecture: 256→dropout→128→dropout→softmax) | 17/21 — **81.0%** |
| XGBoost | 5/21 — **23.8%** |
| VQC (PennyLane) | 15/21 — **71.4%** |

Random Forest and the MLP land close to the original paper's 86.61%
figure, but under an honest per-sample LOOCV instead of whatever split
the original paper used. All four models classify MCT perfectly
(5/5); Normal_Control and Medium_Cyst are the hardest categories
across the board.

## Known issues (not yet resolved — flagging honestly rather than hiding them)

1. **XGBoost has collapsed to predicting a single class.** It predicts
   `Small_Cyst` for all 21 folds regardless of the true label — 100%
   "accuracy" on Small_Cyst samples, 0% on everything else. This is
   after deliberate anti-overfitting tuning (shallow trees, strong L1/L2
   regularization, subsampling), which either wasn't enough or masked a
   different underlying bug (e.g. sample weights not reaching the
   optimizer correctly, or 20 training samples simply being too few
   for gradient boosting to split on anything but the most obvious
   class). Needs targeted debugging before XGBoost numbers are
   reported anywhere — right now they should not be trusted.
2. **Patient-level leakage is unresolved.** GEO's own summary for
   GSE7869 states the 18 cyst/MCT samples came from only 5 kidneys —
   several of the 21 "independent" samples are almost certainly from
   the same patient. Sample-level LOOCV, as used here, does not
   account for that. Needs disclosure as a limitation at minimum, and
   ideally patient-grouped CV if per-sample patient IDs can be
   recovered from the raw GEO metadata.
3. **VQC training accuracy is itself moderate** (0.45–0.85 across
   folds, see `vqc_loocv_fold_results.csv`), meaning the circuit is
   underfitting even its own training data in several folds — likely
   needs more layers/qubits or a different encoding, not just more
   epochs.

## How to run

```bash
# 1. Preprocessing (R) — produces gene_expression_labeled.csv
Rscript preprocess.R

# 2. Classical baseline
pip install mrmr-selection xgboost scikit-learn tensorflow pandas numpy
python phase3_mrmr_loocv_models.py

# 3. Quantum classifier
pip install pennylane
python phase4_vqc_pennylane.py
```

## Next steps

- Debug the XGBoost collapse before using its numbers anywhere.
- Decide on and implement a patient-level CV strategy, or formally
  disclose the leakage risk as a limitation.
- Improve the VQC (encoding strategy, circuit depth) before treating
  71.4% as its ceiling.
- Extend results with per-class precision/recall/F1, not just overall
  accuracy, given the class imbalance.

## Citation

Patil, M., Akkasaligar, P. T., & Pattar, S. *Kidney Cyst Gene
Expression Analysis using Machine Learning.* KLE Technological
University.
