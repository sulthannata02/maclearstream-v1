"""
Sidebar — Navigasi utama aplikasi.

Menu items dibaca dari config.yaml sehingga bisa dikustomisasi
tanpa mengubah code.
"""

from __future__ import annotations

import streamlit as st

from config.settings import get_config, get_ui_config, get_project_config
from src.assets.themes import list_themes
from services.prediction import get_available_models


def render_sidebar() -> tuple[str, str]:
    """
    Render sidebar navigasi dan kembalikan menu yang dipilih.

    Returns
    -------
    tuple[str, str]
        (menu_label_with_emoji, theme_name)
    """
    config = get_config()
    ui_config = get_ui_config()
    project_config = get_project_config()

    # ─── Title ───
    project_name = project_config.get("name", "ML Project")
    page_icon = ui_config.get("page_icon", "🤖")
    st.sidebar.title(f"{page_icon} {project_name}")

    st.sidebar.markdown("---")

    # ─── Navigation Menu ───
    menu_items = ui_config.get("menu", [])

    if not menu_items:
        # Default menu jika config kosong
        menu_items = [
            {"emoji": "🏠", "label": "Beranda", "view": "home"},
            {"emoji": "📊", "label": "Analisis Data", "view": "eda"},
            {"emoji": "🔮", "label": "Prediksi", "view": "prediction"},
            {"emoji": "ℹ️", "label": "Tentang", "view": "about"},
        ]

    menu_options = [
        f"{item['emoji']} {item['label']}" for item in menu_items
    ]

    selected_menu = st.sidebar.radio(
        "Navigasi",
        menu_options,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # ─── Theme Selector ───
    available_themes = list_themes()
    theme_names = [t["name"] for t in available_themes]
    theme_ids = [t["id"] for t in available_themes]

    current_theme = ui_config.get("theme", "dark_premium")
    default_idx = 0
    if current_theme in theme_ids:
        default_idx = theme_ids.index(current_theme)

    selected_theme_name = st.sidebar.selectbox(
        "🎨 Tema",
        theme_names,
        index=default_idx,
    )
    selected_theme_id = theme_ids[theme_names.index(selected_theme_name)]

    st.sidebar.markdown("---")

    # ─── Model Status ───
    trained_models = get_available_models()
    if trained_models:
        st.sidebar.success(f"✅ {len(trained_models)} model tersedia")
    else:
        st.sidebar.warning("⚠️ Belum ada model di-train")

    # ─── Footer ───
    version = project_config.get("version", "1.0.0")
    author = project_config.get("author", "")

    footer_text = f"v{version}"
    if author:
        footer_text += f" • {author}"

    st.sidebar.caption(
        f"ML × Streamlit Framework\n\n{footer_text}"
    )

    return selected_menu, selected_theme_id
