# 🧬 Yeast Dataset — Description

## 📌 Overview
The **Yeast dataset** is a classic multiclass classification dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/110/yeast).  
It originates from a molecular biology study aiming to predict the **subcellular localization of proteins** in yeast cells based on numerical features extracted from protein sequences.

Each instance corresponds to one protein, and the task is to classify it into one of several cellular compartments (e.g., cytoplasm, nucleus, mitochondria).

---

## 📊 Dataset Summary

| Property              | Value            |
|-----------------------|------------------|
| Number of instances   | 1484            |
| Number of attributes  | 8 numerical features (+1 class label) |
| Number of classes     | 10              |
| Task                  | Multiclass classification |
| Missing values        | None            |
| Source                | UCI Machine Learning Repository |

---

## 🧠 Attributes

The dataset contains 8 numerical attributes derived from biological sequence analysis, plus one target class.  
The first column in the original file (`seq_name`) is just a protein identifier and is not used for training.

| # | Name       | Description |
|--:|-----------|-------------|
| 1 | mcg       | McGeoch method for signal sequence recognition |
| 2 | gvh       | von Heijne method for signal sequence recognition |
| 3 | alm       | Score of the ALOM membrane prediction program |
| 4 | mit       | Score of a discriminant analysis for mitochondrial targeting |
| 5 | erl       | Score for ER retention signal |
| 6 | pox       | Peroxisomal targeting signal |
| 7 | vac       | Vacuolar targeting signal |
| 8 | nuc       | Nuclear localization signal |
| - | class     | Target variable: protein localization site |

---

## 🧪 Classes

There are **10 classes**, and the dataset is **imbalanced** — some classes have hundreds of samples, others only a few.  
This imbalance explains why many classifiers struggle to predict the smallest classes (e.g., ERL with 5 samples).

| Class | Description                          | # Samples |
|:-----:|--------------------------------------|----------:|
| CYT   | Cytosol (Cytoplasm)                  | 463 |
| NUC   | Nucleus                              | 429 |
| MIT   | Mitochondria                         | 244 |
| ME3   | Membrane protein, type 3            | 163 |
| ME2   | Membrane protein, type 2            |  51 |
| ME1   | Membrane protein, type 1            |  44 |
| EXC   | Extracellular                        |  37 |
| VAC   | Vacuole                              |  30 |
| POX   | Peroxisome                           |  20 |
| ERL   | Endoplasmic Reticulum Lumen         |   5 |

---

## 📝 Notes

- The dataset has **no missing values**, and all features are continuous.  
- Feature scaling improves the performance of algorithms sensitive to magnitude (e.g., Logistic Regression, k-NN).  
- Macro-averaged metrics are more informative than accuracy because of the class imbalance.

---

## 📚 Reference

Nakai, K. and Kanehisa, M. (1992). *A knowledge base for predicting protein localization sites in eukaryotic cells.*  
Genomics 14: 897–911.


