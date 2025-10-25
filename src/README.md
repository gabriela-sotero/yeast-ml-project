# 🔧 Source Code Directory

This directory contains the core Python modules for the Yeast dataset machine learning project.

## 📁 Structure

```
src/
├── __init__.py           # Package initialization
├── config.py             # Configuration settings and paths
├── data_loader.py        # Data loading utilities
├── preprocessing.py      # Data preprocessing functions
├── models.py             # Machine learning model implementations
├── evaluation.py         # Model evaluation and metrics
├── visualization.py      # Plotting and visualization utilities
└── run_all.py           # Complete pipeline runner
```

## 🧩 Module Descriptions

### **`config.py`** - Configuration Settings
- **Purpose**: Central configuration for the entire project
- **Contents**:
  - Project paths and directories
  - Data file paths
  - Model parameters
  - Random state settings
  - Classifier names
- **Usage**: Imported by all other modules

### **`data_loader.py`** - Data Loading Utilities
- **Purpose**: Load and manage the Yeast dataset
- **Key Functions**:
  - `load_yeast_data()`: Load raw dataset from file
  - `split_data()`: Split data into train/test sets
  - `save_processed_data()`: Save processed data to CSV
  - `load_processed_data()`: Load preprocessed data
- **Features**:
  - Automatic column naming
  - Stratified train/test split
  - Data validation

### **`preprocessing.py`** - Data Preprocessing
- **Purpose**: Clean and prepare data for machine learning
- **Key Functions**:
  - `preprocess_yeast_data()`: Main preprocessing pipeline
  - `scale_features()`: Feature standardization
  - `get_class_distribution()`: Class distribution analysis
- **Features**:
  - Label encoding
  - Feature scaling
  - Data validation

### **`models.py`** - Machine Learning Models
- **Purpose**: Implement and train all classifiers
- **Key Functions**:
  - `get_decision_tree()`: Decision Tree with hyperparameter tuning
  - `get_naive_bayes()`: Naive Bayes classifier
  - `get_logistic_regression()`: Logistic Regression with tuning
  - `get_knn()`: k-Nearest Neighbors with tuning
  - `get_random_forest()`: Random Forest with tuning
  - `train_all_models()`: Train all models at once
- **Features**:
  - GridSearchCV for hyperparameter tuning
  - Automatic model saving
  - Cross-validation

### **`evaluation.py`** - Model Evaluation
- **Purpose**: Evaluate model performance and generate reports
- **Key Functions**:
  - `evaluate_model()`: Evaluate single model
  - `evaluate_all_models()`: Evaluate all models
  - `get_detailed_report()`: Generate detailed classification report
  - `cross_validate_model()`: Cross-validation analysis
- **Metrics**:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC for multi-class problems
  - Confusion matrices

### **`visualization.py`** - Plotting Utilities
- **Purpose**: Create visualizations and plots
- **Key Functions**:
  - `plot_class_distribution()`: Class distribution plots
  - `plot_feature_distributions()`: Feature histograms
  - `plot_correlation_matrix()`: Feature correlation heatmap
  - `plot_confusion_matrix()`: Confusion matrix visualization
  - `plot_model_comparison()`: Model performance comparison
  - `plot_metric_sweep_learning_curves()`: Stratified metric sweep learning curves
- **Features**:
  - High-resolution plots
  - Automatic saving
  - Consistent styling

### **`run_all.py`** - Complete Pipeline
- **Purpose**: Run the entire machine learning pipeline
- **Key Function**:
  - `run_complete_pipeline()`: Execute full analysis
- **Pipeline Steps**:
  1. Load and explore data
  2. Preprocess and split data
  3. Train all models
  4. Evaluate performance
  5. Generate visualizations
  6. Save results

## 🚀 Usage Examples

### Run Complete Pipeline
```python
from src.run_all import run_complete_pipeline
results, models = run_complete_pipeline()
```

### Load and Preprocess Data
```python
from src.data_loader import load_yeast_data
from src.preprocessing import preprocess_yeast_data

data = load_yeast_data()
X, y, label_encoder = preprocess_yeast_data(data)
```

### Train Individual Model
```python
from src.models import get_random_forest
from src.evaluation import evaluate_model

model = get_random_forest(X_train, y_train)
metrics = evaluate_model(model, X_test, y_test, 'random_forest')
```

### Create Visualizations
```python
from src.visualization import plot_class_distribution
plot_class_distribution(y, "Class Distribution", "output.png")
```

## 🔧 Configuration

All settings are centralized in `config.py`:
- **Paths**: Data, results, and model directories
- **Parameters**: Random state, test size, validation size
- **Models**: List of classifiers to train
- **Directories**: Automatic creation of required folders

## 📊 Output Structure

The pipeline generates:
- **Models**: Saved as `.pkl` files in `results/models/`
- **Metrics**: CSV files and text reports in `results/metrics/`
- **Visualizations**: PNG files in `results/figures/`
- **Data**: Processed CSV files in `data/processed/`

## 🎯 Key Features

- **Modular Design**: Each module has a specific purpose
- **Reproducible**: Fixed random seeds and consistent results
- **Configurable**: Easy to modify parameters and settings
- **Comprehensive**: Full ML pipeline from data to results
- **Well-Documented**: Clear function documentation and examples

## 📝 Notes

- All modules use snake_case naming convention
- Functions are well-documented with docstrings
- Error handling for common issues
- Automatic directory creation
- Consistent import structure

