"""
About View — Informasi project.
"""

from __future__ import annotations

import streamlit as st

from config.settings import get_project_config, get_config
from src.layouts.header import render_header
from src.components.cards import info_card, glass_card
from src.components.widgets import key_value_display


def show() -> None:
    """Render halaman Tentang."""
    render_header(
        "Tentang Project",
        "Informasi project dan tech stack",
        emoji="ℹ️",
    )

    project = get_project_config()
    config = get_config()

    # ─── Project Info ───
    glass_card(f"""
        <h2>{project.get('name', 'ML Project')}</h2>
        <p>{project.get('description', 'Machine Learning Project')}</p>
    """)

    st.markdown("### 📋 Project Details")

    key_value_display({
        "Nama": project.get("name", "-"),
        "Versi": project.get("version", "-"),
        "Author": project.get("author", "-"),
        "Task Type": project.get("task_type", "-").title(),
    })

    st.markdown("---")

    # ─── Tech Stack ───
    st.markdown("### 🛠️ Tech Stack")

    cols = st.columns(4)
    with cols[0]:
        info_card("🐍", "Python", "Bahasa pemrograman utama")
    with cols[1]:
        info_card("🎨", "Streamlit", "Framework web UI")
    with cols[2]:
        info_card("🤖", "Scikit-Learn", "Machine Learning library")
    with cols[3]:
        info_card("📊", "Plotly", "Interactive visualization")

    st.markdown("---")

    # ─── Models Info ───
    models = config.get("models", [])
    if models:
        st.markdown("### 🤖 Model yang Dikonfigurasi")

        for m in models:
            with st.expander(f"📦 {m.get('display_name', m.get('name', ''))}"):
                st.markdown(f"**Class:** `{m.get('class', '')}`")
                st.markdown(f"**Task:** `{m.get('task', '')}`")
                st.markdown("**Parameters:**")
                st.json(m.get("params", {}))

    st.markdown("---")

    # ─── Framework Info ───
    st.markdown("### 🚀 ML × Streamlit Framework")

    st.caption(
        "Template ini dibangun dengan arsitektur modular:\n"
        "- **Config-driven** — cukup edit `config.yaml`\n"
        "- **Pipeline Engine** — preprocessing chainable\n"
        "- **Model Registry** — dynamic import dari YAML\n"
        "- **Experiment Tracker** — built-in logging\n"
        "- **Theme Engine** — 4 preset tema\n"
    )
