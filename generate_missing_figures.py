#!/usr/bin/env python3
"""
Script to generate all missing figures by running the analysis directly.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Add src to path
sys.path.append('src')
from data_loader import load_processed_data
from config import *

def create_directories():
    """Create all necessary figure directories."""
    dirs = [
        'results/figures/exploratory',
        'results/figures/decision_tree',
        'results/figures/naive_bayes', 
        'results/figures/logistic_regression',
        'results/figures/knn',
        'results/figures/random_forest',
        'results/figures/comparison'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✅ Created all figure directories")

def generate_exploratory_figures():
    """Generate exploratory data analysis figures."""
    print("📊 Generating exploratory figures...")
    
    # Load data
    from data_loader import load_yeast_data
    data = load_yeast_data()
    
    # Feature names
    feature_names = ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc']
    X = data[feature_names]
    y = data['class']
    
    # 1. Feature distributions
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(feature_names):
        axes[i].hist(X[feature], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].set_title(f'{feature} Distribution', fontsize=12)
        axes[i].set_xlabel(feature, fontsize=10)
        axes[i].set_ylabel('Frequency', fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Feature Distributions', fontsize=16, y=1.02)
    plt.savefig('results/figures/exploratory/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation matrix
    correlation_matrix = X.corr()
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/figures/exploratory/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Generated exploratory figures")

def generate_model_figures():
    """Generate figures for all models."""
    print("🤖 Generating model figures...")
    
    # Load data and models
    X_train, X_test, y_train, y_test = load_processed_data()
    label_encoder = np.load(MODELS_DIR / 'label_encoder.npy', allow_pickle=True)
    class_names = label_encoder
    
    # Load models
    models = {}
    model_names = ['decision_tree', 'naive_bayes', 'logistic_regression', 'knn', 'random_forest']
    
    for name in model_names:
        models[name] = joblib.load(MODELS_DIR / f'{name}.pkl')
    
    # Generate figures for each model
    for name, model in models.items():
        print(f"  📊 Generating figures for {name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{name.replace("_", " ").title()} - Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'results/figures/{name}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Learning Curves
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(12, 6))
        plt.plot(train_sizes, train_mean, 'o-', label='Training Score', color='blue')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.plot(train_sizes, val_mean, 'o-', label='Validation Score', color='red')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.title(f'{name.replace("_", " ").title()} - Learning Curves', fontsize=16)
        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'results/figures/{name}/learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. ROC Curves (for multi-class)
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        n_classes = y_test_bin.shape[1]
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'{name.replace("_", " ").title()} - ROC Curves', fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'results/figures/{name}/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Feature Importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            feature_names = ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc']
            importance = model.feature_importances_
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=True)
            
            plt.figure(figsize=(10, 6))
            bars = plt.barh(importance_df['feature'], importance_df['importance'])
            plt.title(f'{name.replace("_", " ").title()} - Feature Importance', fontsize=16)
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                         f'{width:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            plt.savefig(f'results/figures/{name}/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Decision Tree visualization (for Decision Tree only)
        if name == 'decision_tree':
            plt.figure(figsize=(20, 10))
            plot_tree(model, max_depth=3, feature_names=feature_names, 
                      class_names=class_names, filled=True, rounded=True)
            plt.title('Decision Tree Structure (First 3 Levels)', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'results/figures/{name}/decision_tree_structure.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print("✅ Generated all model figures")

def main():
    """Generate all missing figures."""
    print("🎨 GENERATING ALL MISSING FIGURES")
    print("=" * 40)
    
    # Create directories
    create_directories()
    
    # Generate figures
    generate_exploratory_figures()
    generate_model_figures()
    
    # Check results
    print("\n📊 FINAL FIGURE COUNT:")
    print("=" * 25)
    for root, dirs, files in os.walk("results/figures"):
        for file in files:
            if file.endswith('.png'):
                print(f"  📊 {os.path.join(root, file)}")
    
    print(f"\n✅ All figures generated successfully!")
    print(f"📁 Check results/figures/ for all visualizations")

if __name__ == "__main__":
    main()
