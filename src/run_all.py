"""
Main script to run the complete machine learning pipeline.
"""

import pandas as pd
import numpy as np
from .data_loader import load_yeast_data, split_data, save_processed_data, load_processed_data
from .preprocessing import preprocess_yeast_data, scale_features, get_class_distribution
from .models import train_all_models
from .evaluation import evaluate_all_models, get_detailed_report
from .visualization import (
    plot_class_distribution, plot_feature_distributions, 
    plot_correlation_matrix, plot_model_comparison
)
from .config import *

def run_complete_pipeline():
    """
    Run the complete machine learning pipeline.
    """
    print("=" * 60)
    print("MACHINE LEARNING PIPELINE - YEAST DATASET")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    data = load_yeast_data()
    print(f"Dataset shape: {data.shape}")
    print(f"Features: {list(data.columns[:-1])}")
    print(f"Target classes: {data['class'].unique()}")
    
    # Step 2: Preprocess data
    print("\n2. Preprocessing data...")
    X, y, label_encoder = preprocess_yeast_data(data)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Class distribution:")
    print(get_class_distribution(y))
    
    # Step 3: Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 4: Scale features
    print("\n4. Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 5: Save processed data
    print("\n5. Saving processed data...")
    save_processed_data(
        pd.DataFrame(X_train_scaled, columns=X.columns),
        pd.DataFrame(X_test_scaled, columns=X.columns),
        pd.Series(y_train),
        pd.Series(y_test)
    )
    
    # Step 6: Train models
    print("\n6. Training models...")
    models = train_all_models(X_train_scaled, y_train)
    print(f"Trained {len(models)} models")
    
    # Step 7: Evaluate models
    print("\n7. Evaluating models...")
    results = evaluate_all_models(models, X_test_scaled, y_test)
    print("\nModel Performance:")
    print(results[['model', 'accuracy', 'precision', 'recall', 'f1_score']].round(4))
    
    # Step 8: Generate detailed reports
    print("\n8. Generating detailed reports...")
    for model_name, model in models.items():
        get_detailed_report(model, X_test_scaled, y_test, model_name)
    
    # Step 9: Create visualizations
    print("\n9. Creating visualizations...")
    
    # Class distribution
    plot_class_distribution(
        y, 
        "Yeast Dataset - Class Distribution",
        FIGURES_DIR / "exploratory" / "class_distribution.png"
    )
    
    # Feature distributions
    plot_feature_distributions(
        X,
        "Yeast Dataset - Feature Distributions",
        FIGURES_DIR / "exploratory" / "feature_distributions.png"
    )
    
    # Correlation matrix
    plot_correlation_matrix(
        X,
        "Yeast Dataset - Feature Correlation Matrix",
        FIGURES_DIR / "exploratory" / "correlation_matrix.png"
    )
    
    # Model comparison
    plot_model_comparison(
        results,
        'accuracy',
        FIGURES_DIR / "comparison" / "model_comparison_accuracy.png"
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Metrics saved to: {METRICS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    
    return results, models

if __name__ == "__main__":
    results, models = run_complete_pipeline()

