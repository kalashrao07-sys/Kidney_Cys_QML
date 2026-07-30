# quick sanity check -- run this once, don't ship it
import pandas as pd

df = pd.read_csv("gene_expression_labeled.csv", index_col=0)

# Pick a random sample
sample = df.drop(columns=["label"]).sample(n=1)

sample.to_csv("test_upload.csv")

print("Actual Label:", df.loc[sample.index, "label"].values[0])
print("Sample Index:", sample.index[0])
# now upload test_upload.csv through the Streamlit UI and confirm it predicts
# that sample's TRUE label (it's an easy sanity check, not a real evaluation --
# the model has already seen this exact sample during training)