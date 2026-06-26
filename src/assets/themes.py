"""
Themes — Preset tema visual untuk aplikasi.

Setiap theme adalah dict dengan key warna yang digunakan oleh
``style.py`` untuk generate CSS.
"""

from __future__ import annotations

THEMES: dict[str, dict[str, str]] = {
    "dark_premium": {
        "name": "Dark Premium",
        "bg_primary": "#0E1117",
        "bg_secondary": "#1A1D26",
        "bg_card": "rgba(26, 29, 38, 0.8)",
        "bg_card_hover": "rgba(36, 39, 50, 0.9)",
        "border_card": "rgba(255, 255, 255, 0.08)",
        "text_primary": "#FAFAFA",
        "text_secondary": "#8B8FA3",
        "accent": "#6C63FF",
        "accent_secondary": "#FF6584",
        "accent_gradient": "linear-gradient(135deg, #6C63FF 0%, #FF6584 100%)",
        "success": "#17B890",
        "warning": "#F59E0B",
        "error": "#EF4444",
        "info": "#3B82F6",
        "glassmorphism": "backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);",
        "shadow": "0 8px 32px rgba(0, 0, 0, 0.3)",
        "radius": "16px",
    },
    "ocean_blue": {
        "name": "Ocean Blue",
        "bg_primary": "#0B1426",
        "bg_secondary": "#112240",
        "bg_card": "rgba(17, 34, 64, 0.8)",
        "bg_card_hover": "rgba(23, 42, 76, 0.9)",
        "border_card": "rgba(100, 180, 255, 0.12)",
        "text_primary": "#CCD6F6",
        "text_secondary": "#8892B0",
        "accent": "#64FFDA",
        "accent_secondary": "#57CBF5",
        "accent_gradient": "linear-gradient(135deg, #64FFDA 0%, #57CBF5 100%)",
        "success": "#64FFDA",
        "warning": "#FFD93D",
        "error": "#FF6B6B",
        "info": "#57CBF5",
        "glassmorphism": "backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);",
        "shadow": "0 8px 32px rgba(0, 0, 0, 0.4)",
        "radius": "12px",
    },
    "forest_green": {
        "name": "Forest Green",
        "bg_primary": "#0D1F0D",
        "bg_secondary": "#1A2E1A",
        "bg_card": "rgba(26, 46, 26, 0.8)",
        "bg_card_hover": "rgba(36, 56, 36, 0.9)",
        "border_card": "rgba(100, 200, 100, 0.1)",
        "text_primary": "#E8F5E9",
        "text_secondary": "#A5C9A5",
        "accent": "#66BB6A",
        "accent_secondary": "#AED581",
        "accent_gradient": "linear-gradient(135deg, #66BB6A 0%, #AED581 100%)",
        "success": "#66BB6A",
        "warning": "#FFB74D",
        "error": "#EF5350",
        "info": "#42A5F5",
        "glassmorphism": "backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);",
        "shadow": "0 8px 32px rgba(0, 0, 0, 0.35)",
        "radius": "14px",
    },
    "minimal_light": {
        "name": "Minimal Light",
        "bg_primary": "#FFFFFF",
        "bg_secondary": "#F8F9FA",
        "bg_card": "rgba(255, 255, 255, 0.95)",
        "bg_card_hover": "rgba(248, 249, 250, 1)",
        "border_card": "rgba(0, 0, 0, 0.08)",
        "text_primary": "#1A1A2E",
        "text_secondary": "#6C757D",
        "accent": "#5B4FDB",
        "accent_secondary": "#E84393",
        "accent_gradient": "linear-gradient(135deg, #5B4FDB 0%, #E84393 100%)",
        "success": "#00B894",
        "warning": "#FDCB6E",
        "error": "#D63031",
        "info": "#0984E3",
        "glassmorphism": "backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);",
        "shadow": "0 4px 24px rgba(0, 0, 0, 0.08)",
        "radius": "12px",
    },
}


def get_theme(name: str) -> dict[str, str]:
    """
    Ambil theme berdasarkan nama.

    Falls back ke 'dark_premium' jika nama tidak ditemukan.
    """
    return THEMES.get(name, THEMES["dark_premium"])


def list_themes() -> list[dict[str, str]]:
    """List semua tema yang tersedia."""
    return [
        {"id": tid, "name": theme["name"]}
        for tid, theme in THEMES.items()
    ]
