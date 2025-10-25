"""
Model evaluation utilities for the Yeast dataset classification.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
import joblib
from .config import *


METRIC_FUNCTIONS = {
    "accuracy": accuracy_score,
    "precision_macro": lambda yt, yp: precision_score(
        yt, yp, average="macro", zero_division=0
    ),
    "recall_macro": lambda yt, yp: recall_score(
        yt, yp, average="macro", zero_division=0
    ),
    "f1_macro": lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
}


MODEL_INDUCTIVE_BIAS = {
    "decision_tree": "High-variance, low-bias learner that memorizes training data unless pruned.",
    "random_forest": "Ensemble of decorrelated trees that reduces variance via averaging.",
    "logistic_regression": "Linear decision boundaries with L2/L1 regularization to control variance.",
    "knn": "Instance-based learner where k controls bias-variance; sensitive to neighborhood size.",
    "naive_bayes": "High-bias generative model with strong independence assumptions, generally low variance.",
}


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
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Try to calculate ROC AUC (only for binary classification or with appropriate encoding)
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:  # Binary classification
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:  # Multi-class classification
                roc_auc = roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="weighted"
                )
        else:
            roc_auc = None
    except:
        roc_auc = None

    metrics = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
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
    results_df.to_csv(METRICS_DIR / "all_models.csv", index=False)

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
    with open(METRICS_DIR / f"{model_name}_report.txt", "w") as f:
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
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    return {"mean_score": scores.mean(), "std_score": scores.std(), "scores": scores}


def metric_sweep_curves(model, X, y, metric="f1_macro", n_splits=5):
    """Compute train/test metric sweeps over stratified train/test splits.

    Args:
        model: Base estimator with fit/predict.
        X (np.ndarray or pd.DataFrame): Feature matrix.
        y (np.ndarray or pd.Series): Target vector.
        metric (str): Metric key from METRIC_FUNCTIONS.
        n_splits (int): Number of reshuffled splits per train size.

    Returns:
        pd.DataFrame: Summary with mean/std metrics per train fraction.
    """

    if metric not in METRIC_FUNCTIONS:
        raise ValueError(
            f"Unsupported metric '{metric}'. Available: {list(METRIC_FUNCTIONS)}"
        )

    X_np = np.asarray(X)
    y_np = np.asarray(y)
    scorer = METRIC_FUNCTIONS[metric]
    records = []

    for frac in range(5, 100, 5):
        train_frac = frac / 100.0
        test_frac = 1.0 - train_frac
        splitter = StratifiedShuffleSplit(
            n_splits=n_splits,
            train_size=train_frac,
            test_size=test_frac,
            random_state=RANDOM_STATE,
        )

        train_scores = []
        test_scores = []

        for train_idx, test_idx in splitter.split(X_np, y_np):
            X_train, X_test = X_np[train_idx], X_np[test_idx]
            y_train, y_test = y_np[train_idx], y_np[test_idx]

            estimator = clone(model)
            estimator.fit(X_train, y_train)

            y_train_pred = estimator.predict(X_train)
            y_test_pred = estimator.predict(X_test)

            train_scores.append(scorer(y_train, y_train_pred))
            test_scores.append(scorer(y_test, y_test_pred))

        records.append(
            {
                "train_frac": train_frac,
                "test_frac": test_frac,
                "train_mean": np.mean(train_scores),
                "train_std": np.std(train_scores),
                "test_mean": np.mean(test_scores),
                "test_std": np.std(test_scores),
            }
        )

    return pd.DataFrame(records)


def interpret_metric_sweep(model_name, sweeps, focus_metric="f1_macro"):
    """Generate bias/variance commentary for a model metric sweep.

    Args:
        model_name (str): Name of the model.
        sweeps (dict[str, pd.DataFrame]): Output from metric_sweep_curves grouped by metric.
        focus_metric (str): Metric to base interpretation on (default: f1_macro).

    Returns:
        str: Narrative description of generalization behavior.
    """

    if focus_metric not in sweeps:
        raise ValueError(f"Focus metric '{focus_metric}' not present in sweeps.")

    df = sweeps[focus_metric]
    avg_gap = (df["train_mean"] - df["test_mean"]).mean()
    gap_trend = (df["train_mean"] - df["test_mean"]).iloc[-1] - (
        df["train_mean"] - df["test_mean"]
    ).iloc[0]
    test_start = df["test_mean"].iloc[0]
    test_end = df["test_mean"].iloc[-1]
    test_delta = test_end - test_start

    if avg_gap > 0.12:
        gap_label = "pronounced train-test gap signalling high variance"
    elif avg_gap < 0.04:
        gap_label = "minimal gap suggesting limited variance"
    else:
        gap_label = "moderate gap indicating balanced bias-variance"

    if test_end >= 0.6:
        performance_label = "strong macro-F1 once enough data is available"
    elif test_end >= 0.45:
        performance_label = "solid but improvable macro-F1"
    else:
        performance_label = "struggling to exceed random-baseline macro-F1"

    if test_delta > 0.08:
        data_effect = "benefits markedly from additional training data"
    elif test_delta < -0.05:
        data_effect = "degrades as the train fraction grows, hinting at instability"
    else:
        data_effect = "remains relatively stable across splits"

    bias_comment = MODEL_INDUCTIVE_BIAS.get(
        model_name, "Model inductive bias description unavailable."
    )

    summary = (
        f"Macro-F1 curves show {gap_label}, with test performance {performance_label}."
        f" The model {data_effect} (Δ={test_delta:.3f})."
        f" Inductive bias: {bias_comment}"
    )

    if gap_trend > 0.05:
        summary += (
            " The gap widens with more training data, reinforcing the variance concern."
        )
    elif gap_trend < -0.05:
        summary += " The gap narrows as training size increases, indicating improved regularization."

    return summary
