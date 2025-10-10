# 📓 Notebooks Directory

This directory contains Jupyter notebooks for interactive analysis of the Yeast dataset machine learning project.

## 📁 Structure

```
notebooks/
├── exploratory_data_analysis.ipynb      # Data exploration and EDA
├── decision_tree_classifier.ipynb       # Decision Tree analysis
├── naive_bayes_classifier.ipynb         # Naive Bayes analysis
├── logistic_regression_classifier.ipynb # Logistic Regression analysis
├── knn_classifier.ipynb                 # k-NN analysis
├── random_forest_classifier.ipynb       # Random Forest analysis
└── model_comparison.ipynb               # Compare all models
```

## 🚀 Getting Started

### Option 1: Jupyter Lab (Recommended)
```bash
jupyter lab
```
Then open: `http://localhost:8889/lab`

### Option 2: Jupyter Notebook
```bash
jupyter notebook
```
Then open: `http://localhost:8888`

## 📊 Notebook Descriptions

### 1. **Exploratory Data Analysis** (`exploratory_data_analysis.ipynb`)
- **Purpose**: Understand the dataset structure and characteristics
- **Contents**:
  - Dataset overview and statistics
  - Class distribution analysis
  - Feature distributions and correlations
  - Data quality assessment
  - Missing value analysis

### 2. **Decision Tree Classifier** (`decision_tree_classifier.ipynb`)
- **Purpose**: Implement and analyze Decision Tree model
- **Contents**:
  - Decision Tree implementation
  - Hyperparameter tuning with GridSearchCV
  - Feature importance analysis
  - Tree visualization
  - Performance evaluation

### 3. **Naive Bayes Classifier** (`naive_bayes_classifier.ipynb`)
- **Purpose**: Implement and analyze Naive Bayes model
- **Contents**:
  - Gaussian Naive Bayes implementation
  - Probability analysis
  - Class conditional probabilities
  - Performance evaluation
  - Confusion matrix analysis

### 4. **Logistic Regression Classifier** (`logistic_regression_classifier.ipynb`)
- **Purpose**: Implement and analyze Logistic Regression model
- **Contents**:
  - Logistic Regression implementation
  - Hyperparameter tuning
  - Coefficient analysis
  - Regularization effects
  - Performance evaluation

### 5. **k-NN Classifier** (`knn_classifier.ipynb`)
- **Purpose**: Implement and analyze k-Nearest Neighbors model
- **Contents**:
  - k-NN implementation
  - Optimal k selection
  - Distance metric analysis
  - Weight analysis (uniform vs distance)
  - Performance evaluation

### 6. **Random Forest Classifier** (`random_forest_classifier.ipynb`)
- **Purpose**: Implement and analyze Random Forest model
- **Contents**:
  - Random Forest implementation
  - Hyperparameter tuning
  - Feature importance analysis
  - Tree ensemble analysis
  - Performance evaluation

### 7. **Model Comparison** (`model_comparison.ipynb`)
- **Purpose**: Compare all models and select the best one
- **Contents**:
  - Side-by-side performance comparison
  - ROC curves and AUC analysis
  - Learning curve comparison
  - Cross-validation results
  - Final model selection

## 🔧 Running the Notebooks

### Prerequisites
Make sure you have all dependencies installed:
```bash
pip install -r requirements.txt
```

### Running Individual Notebooks
1. **Start Jupyter**: `jupyter lab` or `jupyter notebook`
2. **Open the notebook** you want to run
3. **Run all cells**: `Cell` → `Run All`
4. **Or run step by step**: `Shift + Enter` for each cell

### Expected Runtime
- **Exploratory Analysis**: ~2-3 minutes
- **Individual Model Notebooks**: ~5-10 minutes each
- **Model Comparison**: ~3-5 minutes

## 📊 Output Files

Each notebook generates:
- **Visualizations**: Saved to `results/figures/`
- **Models**: Saved to `results/models/`
- **Metrics**: Saved to `results/metrics/`

## 🎯 Key Features

- **Interactive Analysis**: Step-by-step exploration
- **Visualizations**: Rich plots and charts
- **Hyperparameter Tuning**: Automated optimization
- **Model Comparison**: Comprehensive evaluation
- **Reproducible**: All results are reproducible

## 📝 Notes

- All notebooks use the same random seed (42) for reproducibility
- Results are automatically saved to the `results/` directory
- Notebooks can be run independently or in sequence
- Each notebook is self-contained with all necessary imports

