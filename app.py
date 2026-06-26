"""
ML × Streamlit Framework — Entry Point

Jalankan aplikasi:
    streamlit run app.py
"""

import streamlit as st

from config.settings import load_config, get_ui_config
from src.assets.style import inject_custom_css
from src.layouts.sidebar import render_sidebar
from src.routes.router import route

# ──────────────────────────────────────────────
# Load Config
# ──────────────────────────────────────────────
config = load_config()
ui = get_ui_config()

# ──────────────────────────────────────────────
# Page Config (harus di atas semua st.* calls)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title=ui.get("page_title", "ML × Streamlit"),
    page_icon=ui.get("page_icon", "🤖"),
    layout=ui.get("layout", "wide"),
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Render App
# ──────────────────────────────────────────────
menu, theme = render_sidebar()
inject_custom_css(theme)
route(menu)
