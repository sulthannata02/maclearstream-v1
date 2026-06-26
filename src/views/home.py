"""
Home View — Dashboard overview.

Menampilkan ringkasan project, statistik dataset, dan status model.
"""

from __future__ import annotations

import streamlit as st

from config.settings import get_config, get_project_config
from src.components.cards import hero_section, stat_card, info_card
from src.layouts.header import render_header
from services.prediction import get_available_models


def show() -> None:
    """Render halaman Beranda."""
    config = get_config()
    project = get_project_config()

    # Hero Section
    hero_section(
        title=project.get("name", "ML Project"),
        description=project.get(
            "description",
            "Machine Learning Project powered by Streamlit",
        ),
    )

    # Quick Stats
    trained_models = get_available_models()
    task_type = project.get("task_type", "classification").title()
    models_config = config.get("models", [])

    cols = st.columns(4)
    with cols[0]:
        stat_card("🤖", len(models_config), "MODEL TERSEDIA")
    with cols[1]:
        stat_card("✅", len(trained_models), "MODEL TRAINED")
    with cols[2]:
        stat_card("🎯", task_type, "TASK TYPE")
    with cols[3]:
        stat_card("📊", project.get("version", "1.0.0"), "VERSION")

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset Info (jika tersedia)
    try:
        from services.data_loader import load_dataframe, get_dataset_info

        df = load_dataframe()
        info = get_dataset_info(df)

        render_header("📁 Dataset Overview", "Ringkasan dataset yang di-load")

        cols = st.columns(4)
        with cols[0]:
            stat_card("📋", f"{info['rows']:,}", "BARIS DATA")
        with cols[1]:
            stat_card("📐", info["columns"], "KOLOM")
        with cols[2]:
            stat_card("🔢", info["features"], "FITUR")
        with cols[3]:
            stat_card("💾", f"{info['memory_mb']} MB", "MEMORY")

        st.markdown("<br>", unsafe_allow_html=True)

        # Preview data
        with st.expander("📋 Preview Data (5 baris pertama)", expanded=False):
            st.dataframe(df.head(), use_container_width=True)

    except Exception:
        st.info(
            "📁 Dataset belum tersedia. Taruh file CSV di folder `dataset/` "
            "dan sesuaikan path di `config.yaml`."
        )

    # Quick Guide
    render_header("🚀 Quick Start Guide", "Cara menggunakan template ini")

    cols = st.columns(3)
    with cols[0]:
        info_card(
            "1️⃣", "Siapkan Dataset",
            "Taruh file CSV di folder dataset/ dan sesuaikan "
            "config.yaml (dataset, features).",
        )
    with cols[1]:
        info_card(
            "2️⃣", "Training Model",
            "Buka halaman Training → pilih model → klik Train. "
            "Atau via CLI: python -m models.train_model",
        )
    with cols[2]:
        info_card(
            "3️⃣", "Prediksi",
            "Setelah model di-train, buka halaman Prediksi "
            "untuk input manual atau upload CSV.",
        )
