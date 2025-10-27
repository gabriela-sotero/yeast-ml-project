# Machine Learning Project - Yeast Dataset Classification

A comprehensive machine learning project using the **Yeast dataset** (UCI) that implements and compares five classifiers: Decision Tree, Naive Bayes, Logistic Regression, k-NN, and Random Forest. The project includes hyperparameter tuning with cross-validation, learning curve analysis, and comprehensive model evaluation.

## 📁 Project Structure

```
yeast-ml-project/
├── README.md
├── requirements.txt
├── environment.yml                  # Conda environment file
├── .gitignore
│
├── notebooks/                       # Jupyter notebooks by stage
│   ├── exploratory_data_analysis.ipynb
│   ├── decision_tree_classifier.ipynb
│   ├── naive_bayes_classifier.ipynb
│   ├── logistic_regression_classifier.ipynb
│   ├── knn_classifier.ipynb
│   ├── random_forest_classifier.ipynb
│   └── model_comparison.ipynb
│
├── src/                             # Reusable code + pipeline
│   ├── __init__.py
│   ├── config.py                    # Configuration settings
│   ├── data_loader.py               # Data loading utilities
│   ├── preprocessing.py             # Data preprocessing
│   ├── models.py                    # Model definitions and training
│   ├── evaluation.py                # Model evaluation metrics
│   ├── visualization.py             # Plotting utilities
│   └── run_all.py                   # Complete pipeline runner
│
├── data/
│   ├── raw/
│   │   ├── yeast.data               # Original dataset
│   │   └── yeast.names              # Dataset metadata
│   └── processed/
│       ├── X_train.csv              # Training features
│       ├── X_test.csv               # Test features
│       ├── y_train.csv              # Training labels
│       └── y_test.csv                # Test labels
│
├── results/
│   ├── figures/
│   │   ├── exploratory/             # EDA visualizations
│   │   ├── decision_tree/           # Decision Tree plots
│   │   ├── naive_bayes/             # Naive Bayes plots
│   │   ├── logistic_regression/     # Logistic Regression plots
│   │   ├── knn/                     # KNN plots
│   │   ├── random_forest/           # Random Forest plots
│   │   └── comparison/              # Model comparison plots
│   ├── models/                      # Trained model files (.pkl)
│   └── metrics/                     # Evaluation metrics (.csv)
│
└── docs/
    ├── dataset-description.md       # Detailed dataset context
    ├── project-instructions.pdf     # Original assignment
    └── report.pdf                   # Final report (to be generated)
```

## 🚀 Quick Start

### Option 1: Using pip
```bash
git clone https://github.com/gabriela-sotero/yeast-ml-project.git
cd machine-learning
pip install -r requirements.txt
```

### Option 2: Using conda
```bash
git clone https://github.com/gabriela-sotero/yeast-ml-project.git
cd machine-learning
conda env create -f environment.yml
conda activate machine-learning
```

## 🧪 Running the Analysis

### Complete Pipeline
Run the entire analysis pipeline:
```python
from src.run_all import run_complete_pipeline
results, models = run_complete_pipeline()
```

### Individual Notebooks
1. **Exploratory Analysis**: `notebooks/exploratory_data_analysis.ipynb`
2. **Decision Tree**: `notebooks/decision_tree_classifier.ipynb`
3. **Naive Bayes**: `notebooks/naive_bayes_classifier.ipynb`
4. **Logistic Regression**: `notebooks/logistic_regression_classifier.ipynb`
5. **k-NN**: `notebooks/knn_classifier.ipynb`
6. **Random Forest**: `notebooks/random_forest_classifier.ipynb`
7. **Model Comparison**: `notebooks/model_comparison.ipynb`

## 📊 Dataset

The **Yeast dataset** contains 1,484 samples with 8 features and 10 classes. For a detailed description, see [docs/dataset-description.md](docs/dataset-description.md).

### Features:
- `mcg`, `gvh`, `alm`, `mit`, `erl`, `pox`, `vac`, `nuc` (8 numerical features)

### Target Classes:
- 10 different yeast protein localization sites

## 🔧 Methodology

1. **Data Preprocessing**: Stratified train-test split (70% train, 30% test)
2. **Feature Scaling**: StandardScaler for algorithms that require it
3. **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
4. **Model Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
5. **Visualization**: Confusion matrices, learning curves, feature importance

## 📈 Classifiers Implemented

| Classifier | Type | Hyperparameters Tuned |
|------------|------|----------------------|
| Decision Tree | Tree-based | max_depth, min_samples_split, min_samples_leaf, criterion |
| Naive Bayes | Probabilistic | None (GaussianNB) |
| Logistic Regression | Linear | C, penalty, solver |
| k-NN | Instance-based | n_neighbors, weights, metric (Euclidean) |
| Random Forest | Ensemble | n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features |

## 📝 Tech Stack

- **Python 3.9+**
- **scikit-learn**: Machine learning algorithms
- **pandas, numpy**: Data manipulation
- **matplotlib, seaborn**: Visualization
- **jupyter**: Interactive notebooks

## 📚 Documentation

- **Project Instructions**: [docs/project-instructions.pdf](docs/project-instructions.pdf)
- **Dataset Description**: [docs/dataset-description.md](docs/dataset-description.md)
- **Configuration**: See `src/config.py` for all project settings

## 🎯 Key Features

- ✅ **Modular Design**: Reusable code in `src/` directory
- ✅ **Comprehensive Evaluation**: Multiple metrics and visualizations
- ✅ **Hyperparameter Tuning**: Automated optimization for all models
- ✅ **Stratified Splitting**: Maintains class distribution
- ✅ **Cross-Validation**: Robust model evaluation
- ✅ **Feature Scaling**: Proper preprocessing for distance-based algorithms
- ✅ **Visualization**: Rich plots for analysis and presentation
- ✅ **Model Persistence**: Save and load trained models
- ✅ **Reproducible**: Fixed random seeds for consistent results

