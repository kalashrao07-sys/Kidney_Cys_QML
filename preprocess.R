# ============================================================
# GSE7869: raw .CEL.gz -> normalized, labeled expression CSV
# STATUS: Tested and working end-to-end (verified on the real
# GSE7869 data -- confirmed 54675 genes x 21 samples, labels
# confirmed against official GEO metadata, distribution
# Small=5 / Medium=5 / MCT=5 / Large=3 / Normal=3, no NAs).
# ============================================================

# ---- STEP 0: one-time package install ----
options(repos = c(CRAN = "https://cloud.r-project.org"))
my_lib <- Sys.getenv("R_LIBS_USER")
dir.create(my_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(my_lib)
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager", lib = my_lib)
if (!require("affy", quietly = TRUE)) BiocManager::install("affy", lib = my_lib, update = FALSE)
if (!require("R.utils", quietly = TRUE)) install.packages("R.utils", lib = my_lib)

# ---- STEP 1: point this at your actual project folder ----
project_root <- "D:/ML_Paper"                              # <-- edit if different
raw_folder <- file.path(project_root, "GSE7869_RAW")        # folder with the 21 .CEL(.gz) files
setwd(raw_folder)

# ---- STEP 2: un-gzip any remaining .gz files ----
# GOTCHA WE HIT: affy::ReadAffy() can read .CEL.gz files directly.
# If you decompress AND keep the .gz originals in the same folder,
# ReadAffy will load each sample TWICE (once as .CEL, once as .CEL.gz)
# and silently report double the sample count (we got 42 instead of 21).
# This step deletes the .gz after decompressing specifically to avoid that.
gz_files <- list.files(pattern = "\\.gz$")
if (length(gz_files) > 0) {
  for (f in gz_files) R.utils::gunzip(f, remove = TRUE, overwrite = TRUE)
}

# ---- STEP 3: load the 21 raw CEL files ----
library(affy)
raw_data <- ReadAffy(celfile.path = ".")
print(raw_data)                                   # must say "number of samples=21"

# ---- STEP 4: RMA normalization ----
# (background correction + quantile normalization + summarization, in one call)
normalized <- rma(raw_data)

# ---- STEP 5: export the raw (unlabeled) matrix ----
expr_matrix <- exprs(normalized)                   # rows = genes, columns = 21 samples
write.csv(expr_matrix, file.path(project_root, "gene_expression.csv"))
print(dim(expr_matrix))                            # must be 54675 x 21

# ---- STEP 6: attach labels ----
# Source: official GEO metadata for GSE7869
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE7869
# (confirmed to exactly match this paper's own Fig. 3 distribution: 5/5/3/5/3)
label_map <- c(
  GSM190858 = "Small_Cyst",  GSM190859 = "Small_Cyst",  GSM190860 = "Small_Cyst",
  GSM190861 = "Small_Cyst",  GSM190862 = "Small_Cyst",
  GSM190863 = "Medium_Cyst", GSM190864 = "Medium_Cyst", GSM190865 = "Medium_Cyst",
  GSM190866 = "Medium_Cyst", GSM190867 = "Medium_Cyst",
  GSM190868 = "Large_Cyst",  GSM190869 = "Large_Cyst",  GSM190870 = "Large_Cyst",
  GSM190871 = "MCT",         GSM190872 = "MCT",         GSM190873 = "MCT",
  GSM190874 = "MCT",         GSM190875 = "MCT",
  GSM190876 = "Normal_Control", GSM190877 = "Normal_Control", GSM190878 = "Normal_Control"
)

sample_ids <- sub("\\.CEL$", "", colnames(expr_matrix))
labels <- label_map[sample_ids]
print(table(labels, useNA = "ifany"))              # gate check: must show 5/5/3/5/3, zero NAs

# ---- STEP 7: flip to samples-as-rows (what sklearn/Keras/PennyLane expect) and save ----
labeled_data <- as.data.frame(t(expr_matrix))
labeled_data$label <- labels
write.csv(labeled_data, file.path(project_root, "gene_expression_labeled.csv"), row.names = TRUE)

cat("\nDone. gene_expression_labeled.csv has", nrow(labeled_data), "samples x",
    ncol(labeled_data) - 1, "genes + 1 label column.\n")