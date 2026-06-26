"""
File Handler — Save dan load artifacts (pickle, JSON).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def save_artifact(obj: Any, path: str | Path) -> Path:
    """
    Simpan objek ke file pickle via joblib.

    Parameters
    ----------
    obj : Any
        Objek yang akan disimpan (model, scaler, dll).
    path : str | Path
        Path file tujuan.

    Returns
    -------
    Path
        Path file yang disimpan.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_artifact(path: str | Path) -> Any:
    """
    Muat objek dari file pickle.

    Raises
    ------
    FileNotFoundError
        Jika file tidak ditemukan.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact tidak ditemukan: {path}")
    return joblib.load(path)


def save_json(data: Any, path: str | Path) -> Path:
    """Simpan data ke file JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    return path


def load_json(path: str | Path) -> Any:
    """Muat data dari file JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File JSON tidak ditemukan: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def artifact_exists(path: str | Path) -> bool:
    """Cek apakah artifact file ada."""
    return Path(path).exists()


def list_artifacts(
    directory: str | Path,
    extension: str = ".pkl",
) -> list[Path]:
    """List semua artifact files di directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    return sorted(dir_path.glob(f"*{extension}"))
