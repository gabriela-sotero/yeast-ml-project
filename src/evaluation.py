"""
Model evaluation utilities for the Yeast dataset classification.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import cross_val_score
import joblib
from .config import *

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a single model and return metrics.
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        model_name (str): Name of the model
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Try to calculate ROC AUC (only for binary classification or with appropriate encoding)
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:  # Binary classification
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:  # Multi-class classification
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        else:
            roc_auc = None
    except:
        roc_auc = None
    
    metrics = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    return metrics

def evaluate_all_models(models, X_test, y_test):
    """
    Evaluate all models and return a DataFrame of metrics.
    
    Args:
        models (dict): Dictionary of trained models
        X_test, y_test: Test data
        
    Returns:
        pd.DataFrame: DataFrame of evaluation metrics
    """
    results = []
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        metrics = evaluate_model(model, X_test, y_test, model_name)
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(METRICS_DIR / 'all_models.csv', index=False)
    
    return results_df

def get_detailed_report(model, X_test, y_test, model_name):
    """
    Get detailed classification report for a model.
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        model_name (str): Name of the model
        
    Returns:
        str: Classification report
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    # Save detailed report
    with open(METRICS_DIR / f'{model_name}_report.txt', 'w') as f:
        f.write(f"Classification Report for {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))
    
    return report

def cross_validate_model(model, X, y, cv=5):
    """
    Perform cross-validation on a model.
    
    Args:
        model: Model to validate
        X, y: Data
        cv (int): Number of folds
        
    Returns:
        dict: Cross-validation results
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    return {
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'scores': scores
    }

