"""
Forms — Dynamic form builder dari config features.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from config.settings import get_features_config


def build_prediction_form(
    features: list[dict[str, Any]] | None = None,
) -> list[float] | None:
    """
    Auto-generate form input prediksi dari config features.

    Mendukung tipe input:
    - number → st.number_input
    - slider → st.slider
    - select → st.selectbox

    Returns
    -------
    list[float] | None
        Nilai input user, atau None jika form belum di-submit.
    """
    if features is None:
        features = get_features_config()

    if not features:
        st.warning(
            "⚠️ Belum ada feature yang didefinisikan di config.yaml. "
            "Tambahkan section 'features' untuk mengaktifkan form prediksi."
        )
        return None

    values: list[float] = []

    # Layout: 2 kolom
    cols_per_row = 2

    for i in range(0, len(features), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(features):
                break

            feat = features[idx]
            name = feat.get("name", f"feature_{idx}")
            label = feat.get("label", name)
            input_type = feat.get("type", "number")
            help_text = feat.get("help", "")

            with col:
                if input_type == "slider":
                    val = st.slider(
                        label,
                        min_value=feat.get("min", 0.0),
                        max_value=feat.get("max", 100.0),
                        value=feat.get("default", 50.0),
                        step=feat.get("step", 1.0),
                        help=help_text,
                        key=f"pred_form_{name}",
                    )
                elif input_type == "select":
                    options = feat.get("options", [])
                    val = st.selectbox(
                        label,
                        options=options,
                        help=help_text,
                        key=f"pred_form_{name}",
                    )
                    # Convert to numeric if possible
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        val = options.index(val) if val in options else 0
                else:  # number
                    val = st.number_input(
                        label,
                        min_value=feat.get("min", 0),
                        max_value=feat.get("max", 99999),
                        value=feat.get("default", 0),
                        step=feat.get("step", 1),
                        help=help_text,
                        key=f"pred_form_{name}",
                    )

                values.append(float(val))

    return values


def build_upload_form() -> pd.DataFrame | None:
    """
    Form upload CSV untuk batch prediction.

    Returns
    -------
    pd.DataFrame | None
        DataFrame dari file upload, atau None jika belum upload.
    """
    uploaded_file = st.file_uploader(
        "Upload file CSV untuk batch prediction",
        type=["csv"],
        help="File CSV harus memiliki kolom yang sesuai dengan features di config.",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File berhasil di-upload: {df.shape[0]} baris, {df.shape[1]} kolom")

            with st.expander("Preview Data (5 baris pertama)", expanded=True):
                st.dataframe(df.head(), use_container_width=True)

            return df
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")
            return None

    return None
