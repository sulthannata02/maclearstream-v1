"""
EDA View — Exploratory Data Analysis.

Auto-profiling: distribusi, korelasi, missing values, outlier, target balance.
"""

from __future__ import annotations

import streamlit as st

from config.settings import get_config
from src.layouts.header import render_header
from src.components.widgets import empty_state
from src.components.charts import (
    plot_correlation_heatmap,
    plot_distribution,
    plot_target_distribution,
)
from services.data_loader import (
    load_dataframe,
    get_dataset_info,
    get_dataset_profile,
    get_missing_values,
    get_descriptive_stats,
)


def show() -> None:
    """Render halaman Exploratory Data Analysis."""
    render_header(
        "Analisis Data Eksploratif",
        "Profiling otomatis dataset — distribusi, korelasi, missing values",
        emoji="📊",
    )

    try:
        df = load_dataframe()
    except Exception as e:
        empty_state("📁", f"Dataset tidak tersedia: {e}")
        return

    config = get_config()
    target = config.get("dataset", {}).get("target_column", "")

    # ─── Tab Layout ───
    tab_overview, tab_dist, tab_corr, tab_missing = st.tabs([
        "📋 Overview",
        "📊 Distribusi",
        "🔗 Korelasi",
        "❓ Missing Values",
    ])

    # ─── Tab: Overview ───
    with tab_overview:
        info = get_dataset_info(df)

        cols = st.columns(4)
        with cols[0]:
            st.metric("Baris", f"{info['rows']:,}")
        with cols[1]:
            st.metric("Kolom", info["columns"])
        with cols[2]:
            st.metric("Numerik", len(info["numeric_cols"]))
        with cols[3]:
            st.metric("Kategorikal", len(info["categorical_cols"]))

        st.markdown("#### Statistik Deskriptif")
        st.dataframe(get_descriptive_stats(df), use_container_width=True)

        st.markdown("#### Data Profiling")
        st.dataframe(get_dataset_profile(df), use_container_width=True)

        # Target distribution
        if target and target in df.columns:
            st.markdown(f"#### Distribusi Target: `{target}`")
            plot_target_distribution(df, target)

    # ─── Tab: Distribusi ───
    with tab_dist:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if not numeric_cols:
            empty_state("📊", "Tidak ada kolom numerik untuk visualisasi.")
        else:
            selected_col = st.selectbox(
                "Pilih kolom",
                numeric_cols,
                key="eda_dist_col",
            )

            show_by_target = False
            if target and target in df.columns:
                show_by_target = st.checkbox(
                    f"Warnai berdasarkan target ({target})",
                    value=True,
                    key="eda_by_target",
                )

            plot_distribution(
                df,
                selected_col,
                target=target if show_by_target else None,
            )

    # ─── Tab: Korelasi ───
    with tab_corr:
        plot_correlation_heatmap(df)

    # ─── Tab: Missing Values ───
    with tab_missing:
        missing = get_missing_values(df)

        if missing.empty:
            st.success("✅ Tidak ada missing values (NaN) terdeteksi!")
        else:
            st.warning(f"⚠️ {len(missing)} kolom memiliki missing values.")
            st.dataframe(missing, use_container_width=True)

        # Cek zero-as-missing
        zero_cols = config.get("dataset", {}).get("zero_columns", [])
        if zero_cols:
            st.markdown("#### Nilai 0 sebagai Missing Value")
            st.caption(
                "Kolom berikut memiliki nilai 0 yang dianggap missing "
                "(didefinisikan di `config.yaml → dataset.zero_columns`):"
            )
            for col in zero_cols:
                if col in df.columns:
                    zero_count = (df[col] == 0).sum()
                    pct = zero_count / len(df) * 100
                    st.markdown(
                        f"- **{col}**: {zero_count} baris ({pct:.1f}%)"
                    )
