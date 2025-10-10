"""
Machine learning models for the Yeast dataset classification.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from .config import *

def get_decision_tree(X_train, y_train, X_val=None, y_val=None):
    """
    Train a Decision Tree classifier with hyperparameter tuning.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (optional)
        
    Returns:
        sklearn.tree.DecisionTreeClassifier: Trained model
    """
    # Hyperparameter grid for Decision Tree
    param_grid = {
        'max_depth': [3, 5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy']
    }
    
    # Use cross-validation for hyperparameter tuning
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Save the model
    joblib.dump(best_model, MODELS_DIR / 'decision_tree.pkl')
    
    return best_model

def get_naive_bayes(X_train, y_train, X_val=None, y_val=None):
    """
    Train a Naive Bayes classifier.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (optional)
        
    Returns:
        sklearn.naive_bayes.GaussianNB: Trained model
    """
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(nb, MODELS_DIR / 'naive_bayes.pkl')
    
    return nb

def get_logistic_regression(X_train, y_train, X_val=None, y_val=None):
    """
    Train a Logistic Regression classifier with hyperparameter tuning.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (optional)
        
    Returns:
        sklearn.linear_model.LogisticRegression: Trained model
    """
    # Hyperparameter grid for Logistic Regression
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'lbfgs']
    }
    
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    grid_search = GridSearchCV(
        lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Save the model
    joblib.dump(best_model, MODELS_DIR / 'logistic_regression.pkl')
    
    return best_model

def get_knn(X_train, y_train, X_val=None, y_val=None):
    """
    Train a k-Nearest Neighbors classifier with hyperparameter tuning.
    Uses Euclidean distance and cross-validation to find optimal k.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (optional)
        
    Returns:
        sklearn.neighbors.KNeighborsClassifier: Trained model
    """
    # Hyperparameter grid for KNN
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean']
    }
    
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(
        knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Save the model
    joblib.dump(best_model, MODELS_DIR / 'knn.pkl')
    
    return best_model

def get_random_forest(X_train, y_train, X_val=None, y_val=None):
    """
    Train a Random Forest classifier with hyperparameter tuning.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (optional)
        
    Returns:
        sklearn.ensemble.RandomForestClassifier: Trained model
    """
    # Hyperparameter grid for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Save the model
    joblib.dump(best_model, MODELS_DIR / 'random_forest.pkl')
    
    return best_model

def train_all_models(X_train, y_train, X_val=None, y_val=None):
    """
    Train all classifiers and return a dictionary of models.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (optional)
        
    Returns:
        dict: Dictionary of trained models
    """
    models = {}
    
    print("Training Decision Tree...")
    models['decision_tree'] = get_decision_tree(X_train, y_train, X_val, y_val)
    
    print("Training Naive Bayes...")
    models['naive_bayes'] = get_naive_bayes(X_train, y_train, X_val, y_val)
    
    print("Training Logistic Regression...")
    models['logistic_regression'] = get_logistic_regression(X_train, y_train, X_val, y_val)
    
    print("Training k-Nearest Neighbors...")
    models['knn'] = get_knn(X_train, y_train, X_val, y_val)
    
    print("Training Random Forest...")
    models['random_forest'] = get_random_forest(X_train, y_train, X_val, y_val)
    
    return models

