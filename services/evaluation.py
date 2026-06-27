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
    roc_curve,
    precision_recall_curve,
    # Regression
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from config.settings import get_config
from utils.constants import CLASSIFICATION, REGRESSION


def get_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame | None:
    """Ambil feature importance dari model tree atau coef dari linear model."""
    actual_model = model
    if hasattr(model, "model"):
        actual_model = model.model

    importances = None
    if hasattr(actual_model, "feature_importances_"):
        importances = actual_model.feature_importances_
    elif hasattr(actual_model, "coef_"):
        coef = actual_model.coef_
        if coef.ndim > 1:
            importances = np.abs(coef[0])
        else:
            importances = np.abs(coef)

    if importances is not None and len(importances) == len(feature_names):
        df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        return df
    return None


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung korelasi antar variabel numerik."""
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        return num_df.corr()
    return pd.DataFrame()


def evaluate_classification(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluasi model classification.

    Returns
    -------
    dict
        accuracy, precision, recall, f1_score, roc_auc,
        confusion_matrix, classification_report, y_pred, y_proba,
        roc_curve, pr_curve, feature_importance.
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
    roc_curve_data = None
    pr_curve_data = None
    if hasattr(model, "predict_proba"):
        y_proba_full = model.predict_proba(X_test)
        if y_proba_full.shape[1] == 2:
            y_proba = y_proba_full[:, 1]
            try:
                result["roc_auc"] = float(roc_auc_score(y_test, y_proba))
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_curve_data = {"fpr": fpr, "tpr": tpr}
                prec, rec, _ = precision_recall_curve(y_test, y_proba)
                pr_curve_data = {"precision": prec, "recall": rec}
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
    result["roc_curve_data"] = roc_curve_data
    result["pr_curve_data"] = pr_curve_data

    if feature_names:
        result["feature_importance"] = get_feature_importance(model, feature_names)

    return result


def evaluate_regression(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluasi model regression.

    Returns
    -------
    dict
        mae, mse, rmse, r2, y_pred, residuals, feature_importance.
    """
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    result: dict[str, Any] = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "y_pred": y_pred,
        "residuals": residuals,
    }

    if feature_names:
        result["feature_importance"] = get_feature_importance(model, feature_names)

    return result


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str | None = None,
    feature_names: list[str] | None = None,
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
    feature_names : list[str], optional
        Daftar nama fitur.

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
        return evaluate_regression(actual_model, X_test, y_test, feature_names)

    return evaluate_classification(actual_model, X_test, y_test, feature_names)


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

