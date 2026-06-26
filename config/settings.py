"""
Settings — Load config.yaml dan sediakan akses global.

Fungsi ``get_config()`` mengembalikan dict konfigurasi yang sudah
di-cache sehingga hanya dibaca sekali dari disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"


# ──────────────────────────────────────────────
# Config Loader
# ──────────────────────────────────────────────

_config_cache: dict[str, Any] | None = None


def load_config(path: Path | None = None) -> dict[str, Any]:
    """
    Load config.yaml dan kembalikan sebagai dict.

    Parameters
    ----------
    path : Path, optional
        Path ke file config. Default: ``config.yaml`` di root project.

    Returns
    -------
    dict
        Konfigurasi project.

    Raises
    ------
    FileNotFoundError
        Jika file config tidak ditemukan.
    """
    global _config_cache

    config_path = path or CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file tidak ditemukan: {config_path}\n"
            "Pastikan file config.yaml ada di root project."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        _config_cache = yaml.safe_load(f)

    # Resolve relative paths terhadap BASE_DIR
    paths = _config_cache.get("paths", {})
    for key in ("dataset_dir", "artifacts_dir", "experiments_dir"):
        if key in paths:
            paths[key] = str(BASE_DIR / paths[key])

    # Resolve dataset path
    dataset = _config_cache.get("dataset", {})
    if "path" in dataset:
        dataset["path"] = str(BASE_DIR / dataset["path"])

    return _config_cache


def get_config() -> dict[str, Any]:
    """
    Kembalikan config yang sudah di-load (cached).

    Jika belum di-load, otomatis load dari default path.
    """
    global _config_cache

    if _config_cache is None:
        load_config()

    return _config_cache  # type: ignore[return-value]


# ──────────────────────────────────────────────
# Helper Accessors
# ──────────────────────────────────────────────

def get_project_config() -> dict[str, Any]:
    """Kembalikan section 'project' dari config."""
    return get_config().get("project", {})


def get_dataset_config() -> dict[str, Any]:
    """Kembalikan section 'dataset' dari config."""
    return get_config().get("dataset", {})


def get_features_config() -> list[dict[str, Any]]:
    """Kembalikan section 'features' dari config."""
    return get_config().get("features", [])


def get_pipeline_config() -> dict[str, Any]:
    """Kembalikan section 'pipeline' dari config."""
    return get_config().get("pipeline", {})


def get_models_config() -> list[dict[str, Any]]:
    """Kembalikan section 'models' dari config."""
    return get_config().get("models", [])


def get_ui_config() -> dict[str, Any]:
    """Kembalikan section 'ui' dari config."""
    return get_config().get("ui", {})


def get_paths_config() -> dict[str, Any]:
    """Kembalikan section 'paths' dari config."""
    return get_config().get("paths", {})
