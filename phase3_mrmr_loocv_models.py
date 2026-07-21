"""
Phase 3: mRMR feature selection + Random Forest / XGBoost / MLP comparison
Evaluated with Leave-One-Out Cross-Validation (LOOCV).

Why LOOCV: with classes as small as 3 samples (Large_Cyst, Normal_Control),
a fixed k-fold can't stratify properly. LOOCV sidesteps that -- every one
of your 21 samples gets to be the test set exactly once.

Why "nested": mRMR feature selection and the class-balance weights are
recomputed from scratch INSIDE every fold, using only that fold's 20
training samples. If you selected features once on all 21 samples up
front, the held-out sample would have quietly influenced which genes
got chosen before it was ever "tested" -- that's leakage, and it's the
most common reason small-sample gene-expression papers get challenged.

Known limitation (disclosed, not fixed here): GEO's own summary for
GSE7869 says the 18 cyst/MCT samples came from just 5 kidneys, so several
samples are almost certainly from the same patient. This script treats
all 21 as independent, which is the same assumption the original paper
made -- worth a line in your limitations section either way.

Install once:
    pip install mrmr-selection xgboost scikit-learn tensorflow pandas numpy
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from mrmr import mrmr_classif
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# Config
# ============================================================
INPUT_CSV = "gene_expression_labeled.csv"   # from the R labeling step
N_FEATURES = 75      # genes mRMR keeps per fold -- try 50/75/100 later and compare
PREFILTER = 3000      # cheap variance pre-filter before mRMR, for speed (also fold-local)
MLP_EPOCHS = 100

# ============================================================
# Load
# ============================================================
df = pd.read_csv(INPUT_CSV, index_col=0)
X = df.drop(columns=["label"]).reset_index(drop=True)   # clean 0..20 index, kept aligned with y throughout
y_raw = df["label"].reset_index(drop=True)

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_
print("Classes:", list(class_names))
print(f"Running Leave-One-Out over {len(X)} samples x {X.shape[1]} genes...\n")

# ============================================================
# Nested Leave-One-Out loop
# ============================================================
loo = LeaveOneOut()
results = {"RandomForest": [], "XGBoost": [], "MLP": []}
fold_log = []

for fold, (train_idx, test_idx) in enumerate(loo.split(X), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ---- feature selection: fit on this fold's training data ONLY ----
    top_var_genes = X_train.var().sort_values(ascending=False).index[:PREFILTER]
    X_train_pf, X_test_pf = X_train[top_var_genes], X_test[top_var_genes]

    y_train_series = pd.Series(y_train, index=X_train_pf.index)  # keep index aligned for mrmr
    selected = mrmr_classif(X=X_train_pf, y=y_train_series, K=N_FEATURES, show_progress=False)
    X_train_sel, X_test_sel = X_train_pf[selected].values, X_test_pf[selected].values

    # ---- class weights: computed on this fold's training labels ONLY ----
    classes, counts = np.unique(y_train, return_counts=True)
    weight_by_class = {c: len(y_train) / (len(classes) * n) for c, n in zip(classes, counts)}
    sample_weight = np.array([weight_by_class[label] for label in y_train])

    # ---- Model A: Random Forest ----
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42)
    rf.fit(X_train_sel, y_train)
    rf_pred = rf.predict(X_test_sel)[0]
    results["RandomForest"].append((y_test[0], rf_pred))

    # ---- Model B: XGBoost ----
    xgb_model = xgb.XGBClassifier(
    n_estimators=100,          # down from 300
    max_depth=2,               # down from default 6 -- shallow trees only
    learning_rate=0.05,        # down from default 0.3
    subsample=0.7,             # row subsampling for variance reduction
    colsample_bytree=0.5,      # feature subsampling per tree
    reg_lambda=5.0,            # much stronger L2
    reg_alpha=1.0,             # add L1 too -- sparsity helps at this n/p ratio
    min_child_weight=3,        # forbid leaves fit to 1-2 samples
    eval_metric="mlogloss",
    random_state=42,
)
    xgb_model.fit(X_train_sel, y_train, sample_weight=sample_weight)
    xgb_pred = xgb_model.predict(X_test_sel)[0]
    results["XGBoost"].append((y_test[0], xgb_pred))

    # ---- Model C: MLP (same architecture as the original paper) ----
    mlp = keras.Sequential([
        layers.Input(shape=(N_FEATURES,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(class_names), activation="softmax"),
    ])
    mlp.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    mlp.fit(X_train_sel, y_train, class_weight=weight_by_class, epochs=MLP_EPOCHS, verbose=0)
    mlp_pred = int(np.argmax(mlp.predict(X_test_sel, verbose=0), axis=1)[0])
    results["MLP"].append((y_test[0], mlp_pred))

    true_label = class_names[y_test[0]]
    print(f"Fold {fold:2d}/21  true={true_label:15s}  "
          f"RF={class_names[rf_pred]:15s} XGB={class_names[xgb_pred]:15s} MLP={class_names[mlp_pred]}")

    fold_log.append({
        "fold": fold, "true_label": true_label,
        "RF_pred": class_names[rf_pred], "XGB_pred": class_names[xgb_pred], "MLP_pred": class_names[mlp_pred],
    })

# ============================================================
# Report + save
# ============================================================
print("\n" + "=" * 60)
for model_name, preds in results.items():
    y_true = [p[0] for p in preds]
    y_pred = [p[1] for p in preds]
    print(f"\n=== {model_name} (Leave-One-Out, 21 folds) ===")
    print(f"Overall accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

pd.DataFrame(fold_log).to_csv("loocv_fold_results.csv", index=False)
print("Per-fold predictions saved to loocv_fold_results.csv -- use this for your results tables.")
