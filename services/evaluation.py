"""
Evaluation Service — Metrik evaluasi untuk classification & regression.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    # Classification
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    # Regression
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from config.settings import get_config
from utils.constants import CLASSIFICATION, REGRESSION


def evaluate_classification(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """
    Evaluasi model classification.

    Returns
    -------
    dict
        accuracy, precision, recall, f1_score, roc_auc,
        confusion_matrix, classification_report, y_pred, y_proba.
    """
    y_pred = model.predict(X_test)

    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(
            y_test, y_pred, average="weighted", zero_division=0,
        )),
        "recall": float(recall_score(
            y_test, y_pred, average="weighted", zero_division=0,
        )),
        "f1_score": float(f1_score(
            y_test, y_pred, average="weighted", zero_division=0,
        )),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0,
        ),
        "y_pred": y_pred,
    }

    # ROC-AUC (hanya untuk binary classification)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba_full = model.predict_proba(X_test)
        if y_proba_full.shape[1] == 2:
            y_proba = y_proba_full[:, 1]
            try:
                result["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            except ValueError:
                result["roc_auc"] = None
        else:
            y_proba = y_proba_full
            try:
                result["roc_auc"] = float(roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="weighted",
                ))
            except ValueError:
                result["roc_auc"] = None
    else:
        result["roc_auc"] = None

    result["y_proba"] = y_proba

    return result


def evaluate_regression(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """
    Evaluasi model regression.

    Returns
    -------
    dict
        mae, mse, rmse, r2, y_pred.
    """
    y_pred = model.predict(X_test)

    return {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "y_pred": y_pred,
    }


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str | None = None,
) -> dict[str, Any]:
    """
    Evaluasi model berdasarkan task type (auto-detect dari config).

    Parameters
    ----------
    model : Any
        Model yang sudah di-train (ModelWrapper atau sklearn model).
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    task_type : str, optional
        'classification' atau 'regression'. Jika None, baca dari config.

    Returns
    -------
    dict
        Metrik evaluasi sesuai task type.
    """
    if task_type is None:
        config = get_config()
        task_type = config.get("project", {}).get("task_type", CLASSIFICATION)

    # Ambil model sklearn dari wrapper jika perlu
    actual_model = model
    if hasattr(model, "model"):
        actual_model = model.model

    if task_type == REGRESSION:
        return evaluate_regression(actual_model, X_test, y_test)

    return evaluate_classification(actual_model, X_test, y_test)


def compare_models(
    results: dict[str, dict[str, Any]],
    task_type: str | None = None,
) -> pd.DataFrame:
    """
    Compare metrik dari beberapa model.

    Parameters
    ----------
    results : dict
        Mapping nama_model → dict metrik.

    Returns
    -------
    pd.DataFrame
        Tabel perbandingan.
    """
    if task_type is None:
        config = get_config()
        task_type = config.get("project", {}).get("task_type", CLASSIFICATION)

    if task_type == REGRESSION:
        metric_keys = ["mae", "mse", "rmse", "r2"]
    else:
        metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    rows = []
    for name, metrics in results.items():
        row: dict[str, Any] = {"Model": name}
        for key in metric_keys:
            val = metrics.get(key)
            if val is not None:
                row[key.upper()] = round(val, 4)
            else:
                row[key.upper()] = "-"
        rows.append(row)

    return pd.DataFrame(rows)
