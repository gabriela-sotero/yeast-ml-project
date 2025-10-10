"""
Data preprocessing utilities for the Yeast dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from .config import *

def preprocess_yeast_data(data):
    """
    Preprocess the Yeast dataset.
    
    Args:
        data (pd.DataFrame): Raw dataset
        
    Returns:
        tuple: (X, y) - Features and target
    """
    # Separate features and target
    X = data.drop('class', axis=1)
    y = data['class']
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Store the label encoder for later use
    np.save(MODELS_DIR / 'label_encoder.npy', label_encoder.classes_)
    
    return X, y_encoded, label_encoder

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    
    Args:
        X_train, X_test (pd.DataFrame): Training and test features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    import joblib
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    
    return X_train_scaled, X_test_scaled, scaler

def get_class_distribution(y):
    """
    Get the distribution of classes in the dataset.
    
    Args:
        y (array-like): Target variable
        
    Returns:
        pd.Series: Class distribution
    """
    return pd.Series(y).value_counts().sort_index()

