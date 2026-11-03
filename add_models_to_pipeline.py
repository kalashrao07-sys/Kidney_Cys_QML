"""
add_models_to_pipeline.py
Trains FINAL RF, XGBoost and MLP models on all 21 samples, using the exact same
selected genes + StandardScaler already stored in deployment_pipeline.joblib.
Saves: rf_final.joblib, xgb_final.joblib, mlp_final.keras
(deployment_pipeline.joblib is NOT modified.)

Run once, from the folder that contains deployment_pipeline.joblib and the CSV:
    pip install xgboost tensorflow --break-system-packages
    python add_models_to_pipeline.py
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from xgboost import XGBClassifier

# ---------------- CONFIG: set these to match your final LOOCV scripts ----------------
DATA_CSV = "gene_expression_labeled.csv"   # samples x probe IDs, label in LAST column
SEED = 42
XGB_PARAMS = dict(n_estimators=200, max_depth=2, learning_rate=0.1,
                  reg_lambda=1.0, objective="multi:softprob",
                  eval_metric="mlogloss", random_state=SEED)   # <-- copy from your LOOCV XGB config
MLP_EPOCHS = 250                            # <-- copy from your LOOCV MLP script
MLP_BATCH = 8                               # <-- copy from your LOOCV MLP script
# --------------------------------------------------------------------------------------

pipe = joblib.load("deployment_pipeline.joblib")
genes, scaler = pipe["selected_genes"], pipe["scaler"]
svm_final, classes = pipe["svm_final"], list(pipe["class_names"])

df = pd.read_csv(DATA_CSV, index_col=0)
lab = df.iloc[:, -1]
y = lab.map({c: i for i, c in enumerate(classes)}) if lab.dtype == object else lab.astype(int)
assert not y.isna().any(), f"Label values {sorted(set(lab))} don't match class_names {classes}"
y = y.values.astype(int)
X = scaler.transform(df[genes].values)

# Sanity check: labels/rows must line up with what the saved SVM was trained on.
agree = (svm_final.predict(X) == y).mean()
print(f"Alignment check: saved SVM agrees with CSV labels on {agree:.0%} of training samples")
if agree < 0.8:
    raise SystemExit("Labels don't line up with the saved pipeline -- fix DATA_CSV/label column first.")

# ---- Random Forest ----
rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=SEED).fit(X, y)
joblib.dump(rf, "rf_final.joblib")
print("RF train acc:", rf.score(X, y))

# ---- XGBoost ----
xgb = XGBClassifier(**XGB_PARAMS).fit(X, y, sample_weight=compute_sample_weight("balanced", y))
joblib.dump(xgb, "xgb_final.joblib")
print("XGB train acc:", xgb.score(X, y))

# ---- MLP (256-128, ReLU, dropout 0.3, softmax) ----
import tensorflow as tf
tf.keras.utils.set_random_seed(SEED)
mlp = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(256, activation="relu"), tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"), tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(classes), activation="softmax"),
])
mlp.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(len(classes)), y=y)))
mlp.fit(X, y, epochs=MLP_EPOCHS, batch_size=MLP_BATCH, class_weight=cw, verbose=0)
mlp.save("mlp_final.keras")
print("MLP train acc:", mlp.evaluate(X, y, verbose=0)[1])
print("Done: rf_final.joblib, xgb_final.joblib, mlp_final.keras")
