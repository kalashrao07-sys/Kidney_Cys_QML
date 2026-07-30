# quick sanity check -- run this once, don't ship it
import pandas as pd, numpy as np
df = pd.read_csv("gene_expression_labeled.csv", index_col=0)
sample = df.drop(columns=["label"]).iloc[[0]]  # take a real training sample as a fake "new upload"
sample.to_csv("test_upload.csv")
# now upload test_upload.csv through the Streamlit UI and confirm it predicts
# that sample's TRUE label (it's an easy sanity check, not a real evaluation --
# the model has already seen this exact sample during training)