"""
Data loading utilities for the Yeast dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from .config import *

def load_yeast_data():
    """
    Load the Yeast dataset from the raw data file.
    
    Returns:
        pd.DataFrame: The loaded dataset with proper column names
    """
    # Column names for the Yeast dataset
    column_names = [
        'seq_name', 'mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc', 'class'
    ]
    
    # Load the data
    data = pd.read_csv(YEAST_DATA_PATH, sep=r'\s+', header=None)

    data.columns = column_names

    data = data.set_index('seq_name')
    
    return data

def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Split data into training and test sets with stratification.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion of data for test set
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test):
    """
    Save processed data to CSV files.
    
    Args:
        X_train, X_test (pd.DataFrame): Training and test features
        y_train, y_test (pd.Series): Training and test targets
    """
    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False)
    
    print(f"Processed data saved to {PROCESSED_DATA_DIR}")

def load_processed_data():
    """
    Load processed training and test data.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()
    
    return X_train, X_test, y_train, y_test

