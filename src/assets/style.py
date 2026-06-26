"""
Style — Custom CSS injection dengan tema dinamis.
"""

from __future__ import annotations

import streamlit as st

from src.assets.themes import get_theme


def inject_custom_css(theme_name: str = "dark_premium") -> None:
    """
    Inject custom CSS berdasarkan tema yang dipilih.

    Parameters
    ----------
    theme_name : str
        Nama tema dari themes.py.
    """
    t = get_theme(theme_name)

    css = f"""
    <style>
    /* ─── Google Fonts ─── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ─── Global ─── */
    .stApp {{
        font-family: 'Inter', sans-serif;
    }}

    /* ─── Cards ─── */
    .card {{
        background: {t['bg_card']};
        {t['glassmorphism']}
        border: 1px solid {t['border_card']};
        border-radius: {t['radius']};
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: {t['shadow']};
        transition: all 0.3s ease;
    }}
    .card:hover {{
        background: {t['bg_card_hover']};
        transform: translateY(-2px);
        box-shadow: {t['shadow']}, 0 4px 12px rgba(0,0,0,0.15);
    }}
    .card h3 {{
        margin: 0 0 0.5rem 0;
        font-weight: 600;
        font-size: 1.1rem;
    }}
    .card p {{
        margin: 0;
        color: {t['text_secondary']};
        font-size: 0.9rem;
        line-height: 1.5;
    }}

    /* ─── Glass Card ─── */
    .glass-card {{
        background: {t['bg_card']};
        {t['glassmorphism']}
        border: 1px solid {t['border_card']};
        border-radius: {t['radius']};
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: {t['shadow']};
    }}

    /* ─── Stat Card ─── */
    .stat-card {{
        background: {t['bg_card']};
        {t['glassmorphism']}
        border: 1px solid {t['border_card']};
        border-radius: {t['radius']};
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
    }}
    .stat-card:hover {{
        transform: translateY(-3px);
        box-shadow: {t['shadow']};
    }}
    .stat-card .stat-icon {{
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }}
    .stat-card .stat-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {t['accent']};
    }}
    .stat-card .stat-label {{
        font-size: 0.8rem;
        color: {t['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.25rem;
    }}

    /* ─── Hero Section ─── */
    .hero {{
        background: {t['accent_gradient']};
        border-radius: {t['radius']};
        padding: 2.5rem;
        margin-bottom: 2rem;
        color: white;
    }}
    .hero h1 {{
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }}
    .hero p {{
        font-size: 1rem;
        opacity: 0.9;
        margin: 0;
    }}

    /* ─── Page Header ─── */
    .page-header {{
        border-bottom: 2px solid {t['border_card']};
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }}
    .page-header h2 {{
        font-weight: 700;
        margin: 0;
    }}
    .page-header .breadcrumb {{
        color: {t['text_secondary']};
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }}

    /* ─── Status Badge ─── */
    .badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}
    .badge-success {{
        background: {t['success']}22;
        color: {t['success']};
        border: 1px solid {t['success']}44;
    }}
    .badge-warning {{
        background: {t['warning']}22;
        color: {t['warning']};
        border: 1px solid {t['warning']}44;
    }}
    .badge-error {{
        background: {t['error']}22;
        color: {t['error']};
        border: 1px solid {t['error']}44;
    }}
    .badge-info {{
        background: {t['info']}22;
        color: {t['info']};
        border: 1px solid {t['info']}44;
    }}

    /* ─── Empty State ─── */
    .empty-state {{
        text-align: center;
        padding: 3rem 2rem;
        color: {t['text_secondary']};
    }}
    .empty-state .empty-icon {{
        font-size: 3rem;
        margin-bottom: 1rem;
    }}
    .empty-state p {{
        font-size: 0.95rem;
    }}

    /* ─── Custom Divider ─── */
    .divider {{
        height: 1px;
        background: {t['border_card']};
        margin: 1.5rem 0;
    }}

    /* ─── Sidebar Branding ─── */
    [data-testid="stSidebar"] {{
        background: {t['bg_secondary']};
    }}

    /* ─── Table Styling ─── */
    .stDataFrame {{
        border-radius: 8px;
        overflow: hidden;
    }}

    /* ─── Button Override ─── */
    .stButton > button {{
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
    }}

    /* ─── Metric Cards ─── */
    [data-testid="stMetric"] {{
        background: {t['bg_card']};
        border: 1px solid {t['border_card']};
        border-radius: 12px;
        padding: 1rem;
    }}

    /* ─── Animations ─── */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .fade-in {{
        animation: fadeIn 0.5s ease-out;
    }}

    @keyframes slideIn {{
        from {{ opacity: 0; transform: translateX(-20px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}
    .slide-in {{
        animation: slideIn 0.4s ease-out;
    }}

    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.7; }}
    }}
    .pulse {{
        animation: pulse 2s ease-in-out infinite;
    }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)
