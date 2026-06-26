"""
Router — Dynamic menu → view dispatcher.

Membaca menu items dari config dan memetakan ke view functions
di ``src/views/``. Tidak perlu hardcode mapping.
"""

from __future__ import annotations

from config.settings import get_ui_config
from src.views import (
    home,
    eda,
    preprocessing,
    training,
    evaluation,
    prediction,
    experiments,
    about,
)


# Mapping view name (dari config) → module view
VIEW_REGISTRY: dict[str, object] = {
    "home": home,
    "eda": eda,
    "preprocessing": preprocessing,
    "training": training,
    "evaluation": evaluation,
    "prediction": prediction,
    "experiments": experiments,
    "about": about,
}


def route(menu: str) -> None:
    """
    Delegasikan tampilan ke view yang sesuai berdasarkan menu.

    Parameters
    ----------
    menu : str
        Label menu yang dipilih (format: "emoji label", e.g., "🏠 Beranda").
    """
    ui_config = get_ui_config()
    menu_items = ui_config.get("menu", [])

    # Cari view name dari menu label
    for item in menu_items:
        full_label = f"{item['emoji']} {item['label']}"
        if full_label == menu:
            view_name = item.get("view", "")
            view_module = VIEW_REGISTRY.get(view_name)

            if view_module and hasattr(view_module, "show"):
                view_module.show()
                return

    # Fallback: home
    home.show()
