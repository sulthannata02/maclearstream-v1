"""
Base Model Wrapper — Interface standar untuk semua model ML.

Setiap model di registry dibungkus oleh ``ModelWrapper`` yang
menyediakan API konsisten untuk train, predict, dan evaluate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import cross_val_score


class ModelWrapper:
    """
    Wrapper generik untuk model sklearn.

    Parameters
    ----------
    name : str
        Nama identifier model (e.g., 'random_forest').
    display_name : str
        Nama tampilan untuk UI (e.g., 'Random Forest').
    model_instance : Any
        Instance model sklearn yang sudah di-instantiate.
    """

    def __init__(
        self,
        name: str,
        display_name: str,
        model_instance: Any,
    ) -> None:
        self.name = name
        self.display_name = display_name
        self.model = model_instance
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Apakah model sudah di-train."""
        return self._is_fitted

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "ModelWrapper":
        """
        Fit model pada training data.

        Returns
        -------
        ModelWrapper
            Self, untuk method chaining.
        """
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediksi label/nilai."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """
        Prediksi probabilitas (classification only).

        Returns None jika model tidak support predict_proba.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = "accuracy",
    ) -> dict[str, Any]:
        """
        Jalankan K-Fold cross-validation.

        Returns
        -------
        dict
            scores (array), mean, std.
        """
        scores = cross_val_score(
            self.model, X, y,
            cv=cv, scoring=scoring,
        )
        return {
            "scores": scores.tolist(),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "cv": cv,
            "scoring": scoring,
        }

    def get_params(self) -> dict[str, Any]:
        """Ambil hyperparameters model."""
        if hasattr(self.model, "get_params"):
            return self.model.get_params()
        return {}

    def get_feature_importances(self) -> np.ndarray | None:
        """
        Ambil feature importances (jika tersedia).

        Mendukung tree-based models dan linear models.
        """
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        if hasattr(self.model, "coef_"):
            return np.abs(self.model.coef_).flatten()
        return None

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"ModelWrapper(name='{self.name}', status={status})"
