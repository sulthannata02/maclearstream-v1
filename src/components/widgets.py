"""
Widgets — Reusable UI widgets: badges, progress, empty state, toast.
"""

from __future__ import annotations

import streamlit as st


def status_badge(text: str, status: str = "info") -> None:
    """
    Render status badge berwarna.

    Parameters
    ----------
    text : str
        Teks badge.
    status : str
        'success', 'warning', 'error', atau 'info'.
    """
    st.markdown(
        f'<span class="badge badge-{status}">{text}</span>',
        unsafe_allow_html=True,
    )


def progress_section(label: str, progress: float) -> None:
    """
    Render progress bar custom.

    Parameters
    ----------
    label : str
        Label di atas progress bar.
    progress : float
        Nilai 0.0 – 1.0.
    """
    percentage = int(progress * 100)
    st.markdown(f"**{label}** — {percentage}%")
    st.progress(progress)


def empty_state(emoji: str, message: str) -> None:
    """Render empty state placeholder."""
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-icon">{emoji}</div>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def divider() -> None:
    """Render custom divider."""
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


def key_value_display(data: dict, columns: int = 2) -> None:
    """Display key-value pairs dalam grid columns."""
    items = list(data.items())

    for i in range(0, len(items), columns):
        cols = st.columns(columns)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(items):
                key, value = items[idx]
                with col:
                    st.markdown(f"**{key}**")
                    st.markdown(f"`{value}`")
