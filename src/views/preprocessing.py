"""
Preprocessing View — Preview pipeline steps dan transformasi.
"""

from __future__ import annotations

import streamlit as st

from config.settings import get_config
from src.layouts.header import render_header
from src.components.widgets import empty_state
from services.preprocessing import get_pipeline_summary
from services.data_loader import load_dataframe


def show() -> None:
    """Render halaman Preprocessing."""
    render_header(
        "Preprocessing Pipeline",
        "Preview pipeline steps dan hasil transformasi",
        emoji="🔧",
    )

    config = get_config()

    # ─── Pipeline Steps ───
    st.markdown("### 📋 Pipeline Steps")
    st.caption(
        "Steps dijalankan berurutan dari atas ke bawah. "
        "Konfigurasi di `config.yaml → pipeline.steps`."
    )

    steps = get_pipeline_summary(config)

    if not steps:
        empty_state("🔧", "Belum ada pipeline steps di config.")
        return

    for i, step in enumerate(steps, 1):
        with st.container():
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"### {i}")
            with col2:
                st.markdown(f"**{step['name']}**")
                st.caption(step["description"])
            st.markdown("---")

    # ─── Preview Transformasi ───
    st.markdown("### 🔍 Preview Transformasi")

    try:
        df = load_dataframe()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Before (Raw Data)")
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Shape: {df.shape[0]} × {df.shape[1]}")

        with col2:
            st.markdown("#### After (Preprocessed)")
            try:
                from services.preprocessing import preprocess
                result = preprocess(config)
                processed_df = result.get("dataframe")

                if processed_df is not None:
                    st.dataframe(
                        processed_df.head(10),
                        use_container_width=True,
                    )
                    st.caption(
                        f"Shape: {processed_df.shape[0]} × "
                        f"{processed_df.shape[1]}"
                    )

                    # Delta info
                    row_diff = df.shape[0] - processed_df.shape[0]
                    col_diff = df.shape[1] - processed_df.shape[1]

                    if row_diff != 0 or col_diff != 0:
                        st.info(
                            f"Δ Baris: {'-' if row_diff > 0 else '+'}"
                            f"{abs(row_diff)}, "
                            f"Δ Kolom: {'-' if col_diff > 0 else '+'}"
                            f"{abs(col_diff)}"
                        )

            except Exception as e:
                st.error(f"Pipeline error: {e}")

        # ─── Feature Names ───
        try:
            from services.preprocessing import preprocess
            result = preprocess(config)
            feature_names = result.get("feature_names", [])

            if feature_names:
                st.markdown("#### 📐 Feature Names (setelah preprocessing)")
                st.code(", ".join(feature_names))

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Training Set", f"{len(result['X_train'])} baris")
                with col2:
                    st.metric("Test Set", f"{len(result['X_test'])} baris")

        except Exception:
            pass

    except Exception as e:
        empty_state("📁", f"Dataset tidak tersedia: {e}")
