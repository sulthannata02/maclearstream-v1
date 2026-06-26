"""
Model Registry — Auto-register dan manage model dari config.

Registry membaca daftar model dari ``config.yaml``, import class
sklearn secara dinamis, dan menyediakan akses via nama.

Contoh:
    >>> registry = ModelRegistry.from_config(config)
    >>> model = registry.get("random_forest")
    >>> model.train(X_train, y_train)
"""

from __future__ import annotations

import importlib
from typing import Any

from config.settings import get_models_config
from core.exceptions import ConfigError
from models.base import ModelWrapper


class ModelRegistry:
    """
    Registry untuk mengelola model ML.

    Setiap model di-register sebagai ``ModelWrapper`` dan bisa
    diakses via ``get(name)``.
    """

    def __init__(self) -> None:
        self._models: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        display_name: str,
        model_class: type,
        params: dict[str, Any] | None = None,
    ) -> None:
        """
        Register model baru ke registry.

        Parameters
        ----------
        name : str
            Identifier unik (e.g., 'random_forest').
        display_name : str
            Nama untuk display di UI.
        model_class : type
            Class model sklearn.
        params : dict, optional
            Default hyperparameters.
        """
        self._models[name] = {
            "display_name": display_name,
            "model_class": model_class,
            "params": params or {},
        }

    def get(
        self,
        name: str,
        override_params: dict[str, Any] | None = None,
    ) -> ModelWrapper:
        """
        Ambil model dari registry sebagai ModelWrapper.

        Parameters
        ----------
        name : str
            Nama model.
        override_params : dict, optional
            Override default params.

        Returns
        -------
        ModelWrapper
            Instance model wrapper yang siap di-train.

        Raises
        ------
        ConfigError
            Jika model tidak ditemukan di registry.
        """
        if name not in self._models:
            available = ", ".join(self._models.keys())
            raise ConfigError(
                f"Model '{name}' tidak ditemukan di registry. "
                f"Model tersedia: {available}"
            )

        entry = self._models[name]
        params = {**entry["params"]}

        if override_params:
            params.update(override_params)

        model_instance = entry["model_class"](**params)

        return ModelWrapper(
            name=name,
            display_name=entry["display_name"],
            model_instance=model_instance,
        )

    def list_models(self) -> list[dict[str, str]]:
        """List semua model yang terdaftar."""
        return [
            {"name": name, "display_name": info["display_name"]}
            for name, info in self._models.items()
        ]

    def list_names(self) -> list[str]:
        """List nama semua model."""
        return list(self._models.keys())

    def get_display_name(self, name: str) -> str:
        """Ambil display name dari nama model."""
        if name in self._models:
            return self._models[name]["display_name"]
        return name

    def __contains__(self, name: str) -> bool:
        return name in self._models

    def __len__(self) -> int:
        return len(self._models)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "ModelRegistry":
        """
        Build registry dari config.

        Membaca section ``models`` dari config dan import class
        sklearn secara dinamis.
        """
        registry = cls()

        if config:
            models_list = config.get("models", [])
        else:
            models_list = get_models_config()

        for model_cfg in models_list:
            name = model_cfg.get("name", "")
            display_name = model_cfg.get("display_name", name)
            class_path = model_cfg.get("class", "")
            params = model_cfg.get("params", {})

            if not class_path:
                raise ConfigError(
                    f"Model '{name}' tidak memiliki 'class' di config."
                )

            # Dynamic import: "sklearn.ensemble.RandomForestClassifier"
            model_class = _import_class(class_path)

            registry.register(
                name=name,
                display_name=display_name,
                model_class=model_class,
                params=params,
            )

        return registry


def _import_class(class_path: str) -> type:
    """
    Import class dari full module path string.

    Contoh: "sklearn.ensemble.RandomForestClassifier"
    → from sklearn.ensemble import RandomForestClassifier
    """
    try:
        parts = class_path.rsplit(".", 1)
        if len(parts) != 2:
            raise ConfigError(
                f"Format class path tidak valid: '{class_path}'. "
                "Gunakan format: 'module.submodule.ClassName'"
            )

        module_path, class_name = parts
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    except ImportError as e:
        raise ConfigError(
            f"Tidak bisa import module: '{class_path}'. "
            f"Pastikan package sudah terinstall. Error: {e}"
        ) from e
    except AttributeError as e:
        raise ConfigError(
            f"Class '{class_path}' tidak ditemukan. Error: {e}"
        ) from e
