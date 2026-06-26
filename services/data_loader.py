"""
Data Loader — Load, validasi, dan profiling dataset.

Semua fungsi yang berhubungan dengan membaca dan memahami dataset
ada di sini. View layer cukup memanggil fungsi-fungsi ini.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import get_config, get_dataset_config
from core.exceptions import DatasetNotFoundError, ValidationError


def load_dataframe(path: str | None = None) -> pd.DataFrame:
    """
    Load dataset dari CSV.

    Parameters
    ----------
    path : str, optional
        Path ke file CSV. Jika None, ambil dari config.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    DatasetNotFoundError
        Jika file tidak ditemukan.
    """
    if path is None:
        dataset_config = get_dataset_config()
        path = dataset_config.get("path", "")

    file_path = Path(path)

    if not file_path.exists():
        raise DatasetNotFoundError(str(file_path))

    return pd.read_csv(file_path)


def get_dataset_info(df: pd.DataFrame) -> dict[str, Any]:
    """
    Kembalikan ringkasan info dataset.

    Returns
    -------
    dict
        rows, columns, features, numeric_cols, categorical_cols,
        memory_usage.
    """
    config = get_config()
    target = config.get("dataset", {}).get("target_column", "")

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "features": df.shape[1] - (1 if target in df.columns else 0),
        "target": target,
        "numeric_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_cols": df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    }


def get_dataset_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-profiling: tipe data, unique count, null %, skew, kurtosis.

    Returns
    -------
    pd.DataFrame
        Tabel profiling per kolom.
    """
    profile_data = []

    for col in df.columns:
        col_data: dict[str, Any] = {
            "Kolom": col,
            "Tipe": str(df[col].dtype),
            "Non-Null": df[col].notna().sum(),
            "Null": df[col].isna().sum(),
            "Null %": round(df[col].isna().mean() * 100, 1),
            "Unique": df[col].nunique(),
        }

        if np.issubdtype(df[col].dtype, np.number):
            col_data["Mean"] = round(df[col].mean(), 2)
            col_data["Std"] = round(df[col].std(), 2)
            col_data["Min"] = round(df[col].min(), 2)
            col_data["Max"] = round(df[col].max(), 2)
            col_data["Skew"] = round(df[col].skew(), 2)
            col_data["Kurtosis"] = round(df[col].kurtosis(), 2)
        else:
            col_data["Mean"] = "-"
            col_data["Std"] = "-"
            col_data["Min"] = "-"
            col_data["Max"] = "-"
            col_data["Skew"] = "-"
            col_data["Kurtosis"] = "-"

        profile_data.append(col_data)

    return pd.DataFrame(profile_data)


def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analisis missing values per kolom.

    Returns
    -------
    pd.DataFrame
        Kolom, Jumlah Missing, Persentase.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        return pd.DataFrame(columns=["Kolom", "Jumlah", "Persentase"])

    result = pd.DataFrame({
        "Kolom": missing.index,
        "Jumlah": missing.values,
        "Persentase": (missing.values / len(df) * 100).round(1),
    })

    return result.sort_values("Jumlah", ascending=False).reset_index(drop=True)


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Kembalikan statistik deskriptif."""
    return df.describe().round(2)


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Kembalikan matriks korelasi kolom numerik."""
    return df.corr(numeric_only=True).round(3)


def get_target_distribution(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> pd.DataFrame:
    """
    Kembalikan distribusi target column.

    Returns
    -------
    pd.DataFrame
        Label, Jumlah, Persentase.
    """
    if target_column is None:
        config = get_dataset_config()
        target_column = config.get("target_column", "")

    if target_column not in df.columns:
        return pd.DataFrame()

    counts = df[target_column].value_counts().sort_index()

    dist = pd.DataFrame({
        "Label": counts.index.astype(str),
        "Jumlah": counts.values,
    })
    dist["Persentase"] = (dist["Jumlah"] / dist["Jumlah"].sum() * 100).round(1)

    return dist


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """
    Validasi dataset terhadap config.

    Returns
    -------
    list[str]
        Daftar pesan error. Kosong jika valid.
    """
    config = get_config()
    dataset_config = config.get("dataset", {})
    target = dataset_config.get("target_column", "")

    errors: list[str] = []

    if df.empty:
        errors.append("Dataset kosong (0 baris).")
        return errors

    if target and target not in df.columns:
        errors.append(
            f"Kolom target '{target}' tidak ditemukan di dataset. "
            f"Kolom tersedia: {df.columns.tolist()}"
        )

    # Cek feature columns dari config features
    features_config = config.get("features", [])
    for feat in features_config:
        feat_name = feat.get("name", "")
        if feat_name and feat_name not in df.columns:
            errors.append(f"Feature '{feat_name}' tidak ditemukan di dataset.")

    return errors
