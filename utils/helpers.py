"""
Helpers — Fungsi utilitas umum.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format float (0–1) ke string persentase."""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 4) -> str:
    """Format angka dengan jumlah desimal tertentu."""
    return f"{value:.{decimals}f}"


def get_timestamp() -> str:
    """Kembalikan timestamp formatted saat ini."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_timestamp_filename() -> str:
    """Kembalikan timestamp yang aman untuk nama file."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def truncate_text(text: str, max_length: int = 100) -> str:
    """Potong teks dan tambahkan ellipsis jika terlalu panjang."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def safe_divide(numerator: float, denominator: float) -> float:
    """Bagi dengan aman, return 0 jika pembagi nol."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def dict_to_display(
    data: dict[str, Any],
    key_label: str = "Parameter",
    value_label: str = "Nilai",
) -> list[dict[str, Any]]:
    """Konversi dict ke list of dicts untuk display di tabel."""
    return [
        {key_label: k, value_label: v}
        for k, v in data.items()
    ]
