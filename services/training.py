"""
Training Service — Train, cross-validate, dan tuning model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from config.settings import get_config
from models.registry import ModelRegistry
from models.base import ModelWrapper
from utils.logger import get_logger

logger = get_logger(__name__)


def get_registry(
    config: dict[str, Any] | None = None,
) -> ModelRegistry:
    """Ambil model registry dari config."""
    if config is None:
        config = get_config()
    return ModelRegistry.from_config(config)


def train_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any] | None = None,
) -> ModelWrapper:
    """
    Train satu model berdasarkan nama.

    Returns
    -------
    ModelWrapper
        Model yang sudah di-train.
    """
    registry = get_registry(config)
    model = registry.get(name)

    logger.info(f"Training model: {model.display_name}...")
    model.train(X_train, y_train)
    logger.info(f"Training selesai: {model.display_name}")

    return model


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, ModelWrapper]:
    """
    Train semua model dari config.

    Returns
    -------
    dict[str, ModelWrapper]
        Mapping nama → model yang sudah di-train.
    """
    registry = get_registry(config)
    trained_models: dict[str, ModelWrapper] = {}

    for model_info in registry.list_models():
        name = model_info["name"]
        model = registry.get(name)

        logger.info(f"Training: {model.display_name}...")
        model.train(X_train, y_train)
        trained_models[name] = model
        logger.info(f"Selesai: {model.display_name}")

    return trained_models


def cross_validate_model(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "accuracy",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Jalankan K-Fold cross-validation untuk satu model.

    Returns
    -------
    dict
        scores, mean, std, cv, scoring.
    """
    registry = get_registry(config)
    model = registry.get(name)

    logger.info(
        f"Cross-validation {model.display_name} "
        f"(cv={cv}, scoring={scoring})..."
    )

    result = model.cross_validate(X, y, cv=cv, scoring=scoring)

    logger.info(
        f"CV Result: mean={result['mean']:.4f} ± {result['std']:.4f}"
    )

    return result


def tune_hyperparameters(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict[str, list],
    method: str = "grid",
    cv: int = 5,
    scoring: str = "accuracy",
    n_iter: int = 20,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Hyperparameter tuning via GridSearch atau RandomizedSearch.

    Parameters
    ----------
    name : str
        Nama model di registry.
    param_grid : dict
        Grid parameter untuk search.
    method : str
        'grid' untuk GridSearchCV, 'random' untuk RandomizedSearchCV.
    cv : int
        Jumlah fold cross-validation.
    scoring : str
        Metrik scoring.
    n_iter : int
        Jumlah iterasi (hanya untuk RandomizedSearch).

    Returns
    -------
    dict
        best_params, best_score, cv_results.
    """
    registry = get_registry(config)
    model = registry.get(name)

    logger.info(
        f"Tuning {model.display_name} ({method}, cv={cv})..."
    )

    if method == "grid":
        search = GridSearchCV(
            model.model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
    else:
        search = RandomizedSearchCV(
            model.model,
            param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=42,
        )

    search.fit(X_train, y_train)

    logger.info(f"Best score: {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")

    return {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "best_estimator": search.best_estimator_,
    }
