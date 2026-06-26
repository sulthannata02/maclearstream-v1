"""
Prediction Service — Single dan batch prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import get_config, get_paths_config
from core.exceptions import ModelNotTrainedError, ValidationError
from utils.file_handler import load_artifact, artifact_exists


def _get_artifact_path(model_name: str) -> Path:
    """Kembalikan path artifact untuk model tertentu."""
    paths = get_paths_config()
    artifacts_dir = Path(paths.get("artifacts_dir", "artifacts"))
    return artifacts_dir / f"{model_name}.pkl"


def _get_scaler_path() -> Path:
    """Kembalikan path scaler artifact."""
    paths = get_paths_config()
    artifacts_dir = Path(paths.get("artifacts_dir", "artifacts"))
    return artifacts_dir / "scaler.pkl"


def _get_features_path() -> Path:
    """Kembalikan path feature names artifact."""
    paths = get_paths_config()
    artifacts_dir = Path(paths.get("artifacts_dir", "artifacts"))
    return artifacts_dir / "feature_names.pkl"


def predict_single(
    model_name: str,
    input_data: list[float],
) -> dict[str, Any]:
    """
    Prediksi untuk satu input.

    Parameters
    ----------
    model_name : str
        Nama model (e.g., 'random_forest').
    input_data : list[float]
        Nilai fitur sesuai urutan feature_names.

    Returns
    -------
    dict
        prediction, probability (jika classification).

    Raises
    ------
    ModelNotTrainedError
        Jika model atau scaler belum tersedia.
    ValidationError
        Jika jumlah fitur tidak sesuai.
    """
    model_path = _get_artifact_path(model_name)
    scaler_path = _get_scaler_path()
    features_path = _get_features_path()

    if not artifact_exists(model_path):
        raise ModelNotTrainedError(model_name)

    model = load_artifact(model_path)
    feature_names = load_artifact(features_path) if artifact_exists(features_path) else None

    # Validasi jumlah fitur
    if feature_names and len(input_data) != len(feature_names):
        raise ValidationError(
            f"Jumlah fitur harus {len(feature_names)}, "
            f"diberikan {len(input_data)}."
        )

    # Buat DataFrame agar nama kolom sesuai
    columns = feature_names or [f"feature_{i}" for i in range(len(input_data))]
    input_df = pd.DataFrame([input_data], columns=columns)

    # Scaling jika scaler tersedia
    if artifact_exists(scaler_path):
        scaler = load_artifact(scaler_path)
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df.values

    # Prediksi
    prediction = model.predict(input_scaled)[0]

    result: dict[str, Any] = {
        "prediction": int(prediction) if isinstance(prediction, (np.integer, int)) else float(prediction),
    }

    # Probabilitas (classification only)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0]
        result["probability"] = proba.tolist()

    return result


def predict_batch(
    model_name: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prediksi batch dari DataFrame.

    Parameters
    ----------
    model_name : str
        Nama model.
    df : pd.DataFrame
        DataFrame dengan kolom sesuai feature_names.

    Returns
    -------
    pd.DataFrame
        DataFrame asli + kolom 'Prediction' (dan 'Probability' jika ada).
    """
    model_path = _get_artifact_path(model_name)
    scaler_path = _get_scaler_path()
    features_path = _get_features_path()

    if not artifact_exists(model_path):
        raise ModelNotTrainedError(model_name)

    model = load_artifact(model_path)

    # Ambil feature names
    if artifact_exists(features_path):
        feature_names = load_artifact(features_path)
    else:
        feature_names = df.columns.tolist()

    # Filter kolom yang ada di feature_names
    available_cols = [c for c in feature_names if c in df.columns]
    input_df = df[available_cols].copy()

    # Scaling
    if artifact_exists(scaler_path):
        scaler = load_artifact(scaler_path)
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df.values

    # Prediksi
    result_df = df.copy()
    result_df["Prediction"] = model.predict(input_scaled)

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(input_scaled)
        if probas.shape[1] == 2:
            result_df["Probability"] = probas[:, 1].round(4)
        else:
            for i in range(probas.shape[1]):
                result_df[f"Prob_Class_{i}"] = probas[:, i].round(4)

    return result_df


def get_available_models() -> list[str]:
    """List model yang sudah di-train (artifact tersedia)."""
    paths = get_paths_config()
    artifacts_dir = Path(paths.get("artifacts_dir", "artifacts"))

    if not artifacts_dir.exists():
        return []

    models = []
    for f in artifacts_dir.glob("*.pkl"):
        if f.stem not in ("scaler", "feature_names"):
            models.append(f.stem)

    return sorted(models)
