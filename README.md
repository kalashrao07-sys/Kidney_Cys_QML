<div align="center">

# 🧬 Kidney Cyst Classification using Classical and Quantum Machine Learning

### Gene Expression-Based Multi-Class Classification using Nested LOOCV, mRMR Feature Selection, SVM, Random Forest, MLP, XGBoost, QSVM & VQC

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![PennyLane](https://img.shields.io/badge/PennyLane-Quantum-purple)
![Qiskit](https://img.shields.io/badge/Qiskit-Quantum-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

# 📌 Overview

Kidney cystic diseases are among the major causes of chronic kidney disorders. Early and accurate identification of different cyst types can significantly improve diagnosis and treatment planning.

This project proposes a **Gene Expression-based Multi-Class Classification System** that combines **Classical Machine Learning** and **Quantum Machine Learning** to classify kidney tissue into five different categories.

Unlike conventional studies, this work performs a comprehensive comparison between multiple classical and quantum classifiers while employing robust feature selection and rigorous validation techniques.

The project also explores the practical applicability of Quantum Machine Learning for biomedical gene-expression analysis.

---

# 🎯 Objectives

- Classify kidney tissue using gene expression profiles.
- Compare Classical Machine Learning with Quantum Machine Learning.
- Reduce dimensionality of genomic data using mRMR.
- Evaluate model robustness using Nested Leave-One-Out Cross Validation.
- Study the effectiveness of Quantum Support Vector Machines and Variational Quantum Classifiers for biomedical classification.

---

# 🧬 Dataset

**Dataset Name**

GEO Accession: **GSE7869**

Source:

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE7869

### Dataset Characteristics

| Property | Value |
|----------|-------|
| Samples | 21 |
| Original Features | 54,675 Genes |
| Classes | 5 |
| Platform | Affymetrix Gene Expression Microarray |
| Format | CEL Files |

---

# 📊 Classification Labels

| Label | Description |
|--------|-------------|
| Small Cyst | Small renal cyst tissue |
| Medium Cyst | Medium renal cyst tissue |
| Large Cyst | Large renal cyst tissue |
| MCT | Minimally Cystic Tissue |
| Normal | Healthy Kidney Tissue |

---

# 🧠 Project Workflow

```
Raw CEL Files
        │
        ▼
RMA Normalization
        │
        ▼
Gene Expression Matrix
        │
        ▼
Variance Filtering
(54,675 → 3,000 genes)
        │
        ▼
mRMR Feature Selection
(3,000 → 75 genes)
        │
        ▼
Nested Leave-One-Out Cross Validation
        │
        ├──────────────┐
        ▼              ▼
 Classical ML      Quantum ML
        │              │
        ▼              ▼
 Performance Comparison
```

---

# ⚙️ Methodology

## 1. Data Preprocessing

The raw Affymetrix CEL files were preprocessed using:

- Robust Multi-array Average (RMA)
- Background correction
- Quantile normalization
- Log₂ transformation
- Probe summarization

---

## 2. Feature Selection

High-dimensional genomic datasets often contain redundant and irrelevant genes.

To reduce dimensionality:

### Step 1

Variance Filtering

```
54,675
↓

3,000 genes
```

### Step 2

Minimum Redundancy Maximum Relevance (mRMR)

```
3,000

↓

75 informative genes
```

This improves:

- Model accuracy
- Computational efficiency
- Feature stability
- Generalization

---

## 3. Validation Strategy

This project uses

### Nested Leave-One-Out Cross Validation (Nested LOOCV)

Unlike traditional train-test split,

Nested LOOCV

- prevents data leakage
- performs unbiased hyperparameter tuning
- provides reliable performance estimation

Outer Loop

→ Model Evaluation

Inner Loop

→ Hyperparameter Optimization

---

# 🤖 Classical Machine Learning Models

The following classifiers were implemented:

- Support Vector Machine (RBF Kernel)
- Random Forest
- Multi-Layer Perceptron
- XGBoost

---

# ⚛️ Quantum Machine Learning Models

Quantum models were implemented using PennyLane.

Implemented Models

- Quantum Support Vector Machine (QSVM)
- Variational Quantum Classifier (VQC)

Quantum Feature Encoding

- Amplitude Embedding

Quantum Backend

- PennyLane Default Qubit Simulator

---

# 📈 Performance

| Model | Accuracy |
|---------|----------|
| SVM (RBF) | **90.48%** |
| QSVM | **90.48%** |
| VQC | **85.71%** |
| Random Forest | **80.95%** |
| MLP | **71.43%** |
| XGBoost | **52.38%** |

---

# 📊 Evaluation Metrics

Performance was evaluated using

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

# 🔬 Key Contributions

✅ Multi-class Kidney Cyst Classification

✅ Comparison of Classical and Quantum Machine Learning

✅ Robust Feature Selection using mRMR

✅ Nested LOOCV Evaluation

✅ Quantum Feature Embedding

✅ Biomedical Gene Expression Analysis

---

# 📁 Repository Structure

```
├── Dataset/
│   ├── CEL Files
│   └── Gene Expression Matrix
│
├── preprocessing/
│   ├── preprocess.R
│   └── feature_selection.py
│
├── models/
│   ├── svm.py
│   ├── random_forest.py
│   ├── mlp.py
│   ├── xgboost.py
│   ├── qsvm.py
│   └── vqc.py
│
├── evaluation/
│   ├── nested_loocv.py
│   ├── metrics.py
│   └── confusion_matrix.py
│
├── deployment/
│   ├── train_production_pipeline.py
│   ├── app.py
│   └── deployment_pipeline.joblib
│
├── results/
│
├── figures/
│
├── README.md
│
└── requirements.txt
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/your-username/your-repository.git

cd your-repository
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

### Train Final Deployment Model

```bash
python train_production_pipeline.py
```

### Launch the Web Application

```bash
streamlit run app.py
```

---

# 🧪 Demo

The deployed application accepts a gene expression profile (CSV format) containing normalized probe expression values and predicts the corresponding kidney tissue class.

Predicted outputs include:

- Small Cyst
- Medium Cyst
- Large Cyst
- MCT
- Normal

---

# 📚 Literature

This work is inspired by recent advances in

- Quantum Machine Learning
- Gene Expression Classification
- Biomarker Discovery
- Feature Selection
- Quantum Bioinformatics

Key references include publications from:

- Springer Nature
- PLOS ONE
- Nature Scientific Reports
- Oxford University Press
- MDPI
- Tech Science Press

---

# 🔮 Future Scope

- Real Quantum Hardware Implementation
- Frozen RMA (fRMA) for Single-Sample Prediction
- Explainable AI (SHAP/LIME)
- Multi-omics Integration
- Larger Clinical Datasets
- Deployment on IBM Quantum Runtime
- Hybrid Quantum Neural Networks

---

# 👨‍💻 Authors

**Kalash Rao**

B.E. Computer Science & Engineering (Artificial Intelligence)

KLE Technological University, Belagavi

---

# 🙏 Acknowledgements

- Gene Expression Omnibus (NCBI)
- Affymetrix
- PennyLane
- Scikit-learn
- XGBoost
- Qiskit
- GEO GSE7869 Dataset

---

# ⭐ If you found this project useful

Please consider giving this repository a ⭐.
