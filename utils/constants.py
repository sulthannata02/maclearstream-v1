"""
Constants — Path dan konstanta project.

Semua path di-resolve dari config.yaml. Jika config belum
di-load, gunakan default values.
"""

from __future__ import annotations

from pathlib import Path

# ──────────────────────────────────────────────
# Base Path
# ──────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────
# Default Directories
# ──────────────────────────────────────────────

DATASET_DIR = BASE_DIR / "dataset"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

# ──────────────────────────────────────────────
# Task Types
# ──────────────────────────────────────────────

CLASSIFICATION = "classification"
REGRESSION = "regression"

VALID_TASK_TYPES = {CLASSIFICATION, REGRESSION}

# ──────────────────────────────────────────────
# Pipeline Strategies
# ──────────────────────────────────────────────

MISSING_VALUE_STRATEGIES = {"median", "mean", "mode", "drop"}
OUTLIER_METHODS = {"iqr", "zscore", "none"}
ENCODING_METHODS = {"label", "onehot", "ordinal", "none"}
SCALING_METHODS = {"standard", "minmax", "robust", "none"}

# ──────────────────────────────────────────────
# UI Defaults
# ──────────────────────────────────────────────

DEFAULT_PAGE_TITLE = "ML × Streamlit"
DEFAULT_PAGE_ICON = "🤖"
DEFAULT_LAYOUT = "wide"
DEFAULT_THEME = "dark_premium"
