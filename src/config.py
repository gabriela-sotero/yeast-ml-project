"""
Configuration settings for the machine learning project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"

# Data file paths
YEAST_DATA_PATH = RAW_DATA_DIR / "yeast.data"
YEAST_NAMES_PATH = RAW_DATA_DIR / "yeast.names"

# Processed data paths
X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train.csv"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.3
VALIDATION_SIZE = 0.2  # From the 70% training+validation split

# Classifier names
CLASSIFIERS = [
    "decision_tree",
    "naive_bayes", 
    "logistic_regression",
    "knn",
    "random_forest"
]

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, MODELS_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

