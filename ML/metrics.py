"""
metrics.py 

"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import sys
from typing import Optional

from sklearn.metrics import (
    # Regression
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    # Classification
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


# ─────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────

class BaseMetrics(ABC):
    """
    Abstract base for all metrics.

    Subclasses must implement compute().
    aggregate() is shared — works for any metrics_df.
    """

    @abstractmethod
    def compute(
        self,
        y_true:  np.ndarray,
        y_pred:  np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Compute metrics for one fold.

        Args:
            y_true  : true labels/values
            y_pred  : predicted labels/values
            y_proba : predicted probabilities (classification only)

        Returns:
            dict of metric_name → value
        """
        pass

    def aggregate(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate metrics across folds.
        Computes mean ± std per metric per (target, eval_set).

        Args:
            metrics_df : DataFrame from MLManager
                         must have columns: target_column, evaluation_set, fold

        Returns:
            DataFrame with fold="aggregated", mean and std columns added
        """

        exclude = {
            "target_column", "split_column", "model_type",
            "task_type", "param_mode", "evaluation_set", "fold",
            "n_train", "n_eval", "regularization", "alpha_mode", "best_alpha", "best_C", "best_l1_ratio"
        }
        metric_cols = [
            c for c in metrics_df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(metrics_df[c])
        ]

        rows = []
        for (target, eval_set), grp in metrics_df.groupby(
            ["target_column", "evaluation_set"]
        ):
            if len(grp) < 2:
                continue

            agg_row = {
                "target_column":  target,
                "split_column":   grp["split_column"].iloc[0],
                "model_type":     grp["model_type"].iloc[0],
                "task_type":      grp["task_type"].iloc[0] if "task_type" in grp.columns else "",
                "evaluation_set": eval_set,
                "fold":           "aggregated",
                "n_train":        grp["n_train"].mean(),
                "n_eval":         grp["n_eval"].mean(),
            }

            for col in metric_cols:
                vals = grp[col].dropna()
                agg_row[col]              = vals.mean()
                agg_row[f"{col}_std"]     = vals.std()
                agg_row[f"{col}_min"]     = vals.min()
                agg_row[f"{col}_max"]     = vals.max()

            rows.append(agg_row)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Regression metrics
# ─────────────────────────────────────────────────────────────────────

class RegressionMetrics(BaseMetrics):
    """
    Metrics for regression tasks.

    Computed per fold:
        R²    : coefficient of determination
        RMSE  : root mean squared error
        MSE   : mean squared error
        MAE   : mean absolute error

    Aggregated across folds (if K-Fold / LOOCV):
        mean ± std for each metric
    """

    def compute(
        self,
        y_true:  np.ndarray,
        y_pred:  np.ndarray,
        y_proba: Optional[np.ndarray] = None,  # unused for regression
    ) -> dict:
        """
        Compute regression metrics for one fold.

        Args:
            y_true : array of true continuous values
            y_pred : array of predicted values

        Returns:
            dict with r2, rmse, mse, mae
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(y_true) < 2:
            return {"r2": np.nan, "rmse": np.nan, "mse": np.nan, "mae": np.nan}

        mse = float(mean_squared_error(y_true, y_pred))

        return {
            "r2":   float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mse)),
            "mse":  mse,
            "mae":  float(mean_absolute_error(y_true, y_pred)),
        }


# ─────────────────────────────────────────────────────────────────────
# Classification metrics
# ─────────────────────────────────────────────────────────────────────

class ClassificationMetrics(BaseMetrics):
    """
    Metrics for classification tasks.

    Computed per fold:
        accuracy    : fraction of correct predictions
        sensitivity : true positive rate (recall for positive class)
        specificity : true negative rate
        precision   : positive predictive value
        recall      : same as sensitivity
        f1          : harmonic mean of precision and recall
        auc_roc     : area under ROC curve (requires y_proba)

    Binary vs multiclass:
        Binary     → all metrics computed directly
        Multiclass → macro averaging for f1, precision, recall
                     auc_roc uses ovr (one-vs-rest) strategy

    Aggregated across folds (if K-Fold / LOOCV):
        mean ± std for each metric
    """

    def __init__(self, average: str = "macro"):
        """
        Args:
            average : averaging strategy for multiclass
                      "macro"   → unweighted mean across classes
                      "weighted"→ weighted by class support
                      "binary"  → binary classification only
        """
        self.average = average

    def compute(
        self,
        y_true:  np.ndarray,
        y_pred:  np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Compute classification metrics for one fold.

        Args:
            y_true  : array of true class labels
            y_pred  : array of predicted class labels
            y_proba : array of predicted probabilities
                      shape (n_samples,) for binary
                      shape (n_samples, n_classes) for multiclass

        Returns:
            dict with accuracy, sensitivity, specificity,
                       precision, recall, f1, auc_roc
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        n_classes = len(np.unique(y_true))
        is_binary = (n_classes == 2)

        # Metrics de base 
        accuracy = float(accuracy_score(y_true, y_pred))

        avg = self.average

        precision = float(precision_score(
            y_true, y_pred, average=avg, zero_division=0
        ))
        recall = float(recall_score(
            y_true, y_pred, average=avg, zero_division=0
        ))
        f1 = float(f1_score(
            y_true, y_pred, average=avg, zero_division=0
        ))

        # Sensitivity / Specificity  
        if is_binary:
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
                specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
            except ValueError:
                sensitivity = np.nan
                specificity = np.nan
        else:
            
            sensitivity = recall
            specificity = np.nan

        # AUC-ROC 
        auc_roc = np.nan
        if y_proba is not None:
            try:
                if is_binary:
                    # y_proba can be (n,) or (n, 2)
                    proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                    auc_roc = float(roc_auc_score(y_true, proba))
                else:
                    auc_roc = float(roc_auc_score(
                        y_true, y_proba,
                        multi_class="ovr",
                        average=self.average
                    ))
            except Exception:
                auc_roc = np.nan

        return {
            "accuracy":    accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision":   precision,
            "recall":      recall,
            "f1":          f1,
            "auc_roc":     auc_roc,
        }


# ─────────────────────────────────────────────────────────────────────
# Fonction utilisée by MLManager
# ─────────────────────────────────────────────────────────────────────

def get_metrics(task_type: str, **kwargs) -> BaseMetrics:
    """
    Return the right metrics object for the task type.

    Args:
        task_type : "regression" or "classification"
        **kwargs  : passed to ClassificationMetrics (e.g. average="macro")

    Returns:
        RegressionMetrics or ClassificationMetrics instance

    Usage:
        metrics = get_metrics("regression")
        metrics = get_metrics("classification", average="macro")
    """
    if task_type == "regression":
        return RegressionMetrics()

    if task_type == "classification":
        return ClassificationMetrics(**kwargs)

    sys.exit(
        f"ERREUR : task_type '{task_type}' inconnu.\n"
        f"Valeurs acceptées : 'regression', 'classification'"
    )
