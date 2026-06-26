"""
Header — Page header dengan gradient accent dan breadcrumb.
"""

from __future__ import annotations

import streamlit as st


def render_header(
    title: str,
    description: str = "",
    emoji: str = "",
) -> None:
    """
    Render page header.

    Parameters
    ----------
    title : str
        Judul halaman.
    description : str, optional
        Deskripsi singkat.
    emoji : str, optional
        Emoji di depan judul.
    """
    display_title = f"{emoji} {title}" if emoji else title

    st.markdown(
        f"""
        <div class="page-header fade-in">
            <h2>{display_title}</h2>
            <div class="breadcrumb">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
