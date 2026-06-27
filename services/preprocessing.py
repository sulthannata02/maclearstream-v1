"""
Preprocessing Service — Build dan jalankan pipeline dari config.

Menyediakan interface tinggi yang digunakan oleh UI dan CLI
untuk memproses dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import joblib

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
        Berisi: dataframe, feature_names, scaler, encoders,
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

    # Persist artifacts (scaler, encoders, feature metadata)
    paths_config = config.get("paths", {})
    preprocessors_dir = Path(paths_config.get("preprocessors_dir", "artifacts/preprocessors"))
    encoders_dir = Path(paths_config.get("encoders_dir", "artifacts/encoders"))

    preprocessors_dir.mkdir(parents=True, exist_ok=True)
    encoders_dir.mkdir(parents=True, exist_ok=True)

    if result.get("scaler"):
        joblib.dump(result["scaler"], preprocessors_dir / "scaler.joblib")

    if result.get("encoders"):
        joblib.dump(result["encoders"], encoders_dir / "encoders.joblib")

    # Simpan feature metadata
    with open(preprocessors_dir / "feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"feature_names": result.get("feature_names", [])}, f, indent=2)

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

