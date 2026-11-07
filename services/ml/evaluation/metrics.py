"""
CivicPulse AI - Model Evaluation & Acceptance Metrics

Implements evaluation calculators for Regression, Binary Classification,
Multiclass Classification, Clustering, and Time-Series Forecasting.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    silhouette_score,
)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates MAE, RMSE, and R2 score for repair cost/duration regression."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
    }


def evaluate_binary_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Calculates accuracy, precision, recall, and F1 for urgency predictions."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
    }


def evaluate_multiclass_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculates Macro F1, Macro Precision, Macro Recall, and per-class metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "confusion_matrix": cm,
    }


def evaluate_clustering(X: np.ndarray, labels: np.ndarray, centroids: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Calculates SSE (Inertia) and Silhouette score for geospatial clusters."""
    metrics: Dict[str, float] = {}

    if len(set(labels)) > 1:
        metrics["silhouette_score"] = round(float(silhouette_score(X, labels)), 4)
    else:
        metrics["silhouette_score"] = 0.0

    if centroids is not None:
        sse = 0.0
        for i, point in enumerate(X):
            centroid = centroids[labels[i]]
            sse += float(np.sum((point - centroid) ** 2))
        metrics["sse"] = round(sse, 4)

    return metrics


def evaluate_forecasting(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates MAE, RMSE, and MAPE for time-series forecasts."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))

    # Avoid division by zero in MAPE
    non_zero = y_true != 0
    if np.any(non_zero):
        mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])))
    else:
        mape = 0.0

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
    }
