"""
Visualization utilities for the Yeast dataset classification project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from .config import *
from .evaluation import metric_sweep_curves

# Set style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def plot_class_distribution(y, title="Class Distribution", save_path=None):
    """
    Plot the distribution of classes in the dataset.

    Args:
        y (array-like): Target variable
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    class_counts = pd.Series(y).value_counts().sort_index()

    bars = plt.bar(range(len(class_counts)), class_counts.values)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(range(len(class_counts)), class_counts.index, rotation=45)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_feature_distributions(X, title="Feature Distributions", save_path=None):
    """
    Plot distributions of all features.

    Args:
        X (pd.DataFrame): Features
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    n_features = len(X.columns)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes

    for i, column in enumerate(X.columns):
        if i < len(axes):
            axes[i].hist(X[column], bins=30, alpha=0.7, edgecolor="black")
            axes[i].set_title(f"Distribution of {column}")
            axes[i].set_xlabel(column)
            axes[i].set_ylabel("Frequency")

    # Hide empty subplots
    for i in range(len(X.columns), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_correlation_matrix(X, title="Feature Correlation Matrix", save_path=None):
    """
    Plot correlation matrix of features.

    Args:
        X (pd.DataFrame): Features
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    correlation_matrix = X.corr()

    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
    )
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name, class_names=None, save_path=None):
    """
    Plot confusion matrix for a model.

    Args:
        y_true, y_pred: True and predicted labels
        model_name (str): Name of the model
        class_names (list): Names of classes
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_model_comparison(results_df, metric="accuracy", save_path=None):
    """
    Plot comparison of models based on a metric.

    Args:
        results_df (pd.DataFrame): Results dataframe
        metric (str): Metric to plot
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results_df["model"], results_df[metric])
    plt.title(f"Model Comparison - {metric.title()}")
    plt.xlabel("Model")
    plt.ylabel(metric.title())
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_metric_sweep_learning_curves(
    model, X, y, model_name, metrics=None, base_dir=None
):
    """Plot metric sweeps for varying stratified train/test splits.

    Args:
        model: Estimator configured with best hyperparameters.
        X, y: Features and target arrays (numpy or pandas).
        model_name (str): Model identifier used for figure naming.
        metrics (Iterable[str]): Metrics to sweep; defaults to precision, recall, F1.
        base_dir (Path or str): Output directory; defaults to figures/model_name.

    Returns:
        dict[str, pd.DataFrame]: Metric name to sweep dataframe.
    """

    if metrics is None:
        metrics = ["precision_macro", "recall_macro", "f1_macro"]

    output_dir = Path(base_dir) if base_dir else FIGURES_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframes = {}

    for metric in metrics:
        df = metric_sweep_curves(model, X, y, metric=metric)
        dataframes[metric] = df

        plt.figure(figsize=(10, 6))
        plt.plot(df["train_frac"], df["train_mean"], marker="o", label="Train")
        plt.fill_between(
            df["train_frac"],
            df["train_mean"] - df["train_std"],
            df["train_mean"] + df["train_std"],
            alpha=0.1,
        )
        plt.plot(df["train_frac"], df["test_mean"], marker="o", label="Test")
        plt.fill_between(
            df["train_frac"],
            df["test_mean"] - df["test_std"],
            df["test_mean"] + df["test_std"],
            alpha=0.1,
        )

        metric_title = metric.replace("_", " ").title()
        plt.title(f"{metric_title} Sweep - {model_name}")
        plt.xlabel("Training Fraction")
        plt.ylabel(metric_title)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = output_dir / f"learning_curves_{metric}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    return dataframes
