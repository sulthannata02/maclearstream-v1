"""
Preprocessing Service — Build dan jalankan pipeline dari config.

Menyediakan interface tinggi yang digunakan oleh UI dan CLI
untuk memproses dataset.
"""

from __future__ import annotations

from typing import Any

from config.settings import get_config
from core.pipeline import Pipeline
from services.data_loader import load_dataframe


def preprocess(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Load dataset, build pipeline dari config, dan jalankan.

    Parameters
    ----------
    config : dict, optional
        Full config. Jika None, ambil dari settings.

    Returns
    -------
    dict
        Berisi: dataframe, feature_names, scaler,
        X_train, X_test, y_train, y_test.
    """
    if config is None:
        config = get_config()

    dataset_config = config.get("dataset", {})
    target_column = dataset_config.get("target_column", "target")

    # Load dataframe
    df = load_dataframe()

    # Build pipeline dari config
    pipeline = Pipeline.from_config(config)

    # Jalankan pipeline
    result = pipeline.run(df, target_column=target_column)

    return result


def get_pipeline_summary(
    config: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """
    Kembalikan deskripsi human-readable dari pipeline steps.

    Digunakan oleh UI untuk menampilkan step-step preprocessing.
    """
    if config is None:
        config = get_config()

    pipeline = Pipeline.from_config(config)
    return pipeline.describe()
