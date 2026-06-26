"""
Cards — Reusable card components untuk UI.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def info_card(emoji: str, title: str, description: str) -> None:
    """Render card informatif dengan emoji, judul, dan deskripsi."""
    st.markdown(
        f"""
        <div class="card fade-in">
            <h3>{emoji} {title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def glass_card(content: str) -> None:
    """Render glassmorphism card dengan content HTML."""
    st.markdown(
        f"""
        <div class="glass-card fade-in">
            {content}
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(
    icon: str,
    value: str | int | float,
    label: str,
) -> None:
    """Render stat card dengan icon, value besar, dan label."""
    st.markdown(
        f"""
        <div class="stat-card fade-in">
            <div class="stat-icon">{icon}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row(metrics: list[dict[str, Any]]) -> None:
    """
    Render baris metrik dalam kolom.

    Parameters
    ----------
    metrics : list[dict]
        Setiap dict harus memiliki key 'label' dan 'value'.
        Opsional: 'delta', 'help'.
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.metric(
                label=m["label"],
                value=m["value"],
                delta=m.get("delta"),
                help=m.get("help"),
            )


def section_header(title: str, description: str = "") -> None:
    """Render header section dengan garis bawah."""
    st.markdown(f"### {title}")
    if description:
        st.caption(description)


def hero_section(title: str, description: str) -> None:
    """Render hero section dengan gradient background."""
    st.markdown(
        f"""
        <div class="hero fade-in">
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
