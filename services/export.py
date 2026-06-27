"""
Export Service — Export data untuk download di Streamlit.

Semua fungsi return bytes yang bisa langsung dipakai
oleh ``st.download_button(data=...)``.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd  # type: ignore # pyright: ignore[reportMissingImports]


def export_dataframe_csv(df: pd.DataFrame) -> bytes:
    """Export DataFrame ke CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def export_metrics_json(metrics: dict[str, Any]) -> bytes:
    """Export metrik ke JSON bytes."""
    # Filter out non-serializable values (numpy arrays, etc.)
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            clean[k] = v
        elif isinstance(v, list):
            clean[k] = v

    return json.dumps(
        clean, indent=2, ensure_ascii=False, default=str,
    ).encode("utf-8")


def export_comparison_csv(
    comparison_df: pd.DataFrame,
) -> bytes:
    """Export tabel perbandingan model ke CSV bytes."""
    return comparison_df.to_csv(index=False).encode("utf-8")


def export_predictions_csv(
    predictions_df: pd.DataFrame,
) -> bytes:
    """Export hasil prediksi batch ke CSV bytes."""
    return predictions_df.to_csv(index=False).encode("utf-8")


def export_model_onnx(model: Any, input_shape: int) -> bytes | None:
    """Export sklearn model ke ONNX bytes."""
    try:
        from skl2onnx import convert_sklearn  # type: ignore # pyright: ignore[reportMissingImports]
        from skl2onnx.common.data_types import FloatTensorType  # type: ignore # pyright: ignore[reportMissingImports]

        actual_model = model
        if hasattr(model, "model"):
            actual_model = model.model

        initial_type = [('float_input', FloatTensorType([None, input_shape]))]
        onx = convert_sklearn(actual_model, initial_types=initial_type)
        return onx.SerializeToString()
    except Exception:
        return None


