"""
Logger — Structured logging untuk terminal dan file.

Menyediakan colored output di terminal dan opsional log ke file.

Contoh:
    >>> from utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Training dimulai...")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


# ──────────────────────────────────────────────
# Color Codes (ANSI)
# ──────────────────────────────────────────────

class Colors:
    """ANSI color codes untuk terminal."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BG_RED = "\033[41m"


# ──────────────────────────────────────────────
# Custom Formatter
# ──────────────────────────────────────────────

LEVEL_COLORS = {
    "DEBUG": Colors.DIM + Colors.WHITE,
    "INFO": Colors.GREEN,
    "WARNING": Colors.YELLOW,
    "ERROR": Colors.RED,
    "CRITICAL": Colors.BG_RED + Colors.WHITE,
}


class ColoredFormatter(logging.Formatter):
    """Formatter dengan warna ANSI untuk terminal."""

    FORMAT = (
        "{color}[{levelname:<8}]{reset} "
        "{dim}{asctime}{reset} "
        "{cyan}{name}{reset} → "
        "{message}"
    )

    def format(self, record: logging.LogRecord) -> str:
        color = LEVEL_COLORS.get(record.levelname, Colors.WHITE)

        formatted = self.FORMAT.format(
            color=color,
            reset=Colors.RESET,
            dim=Colors.DIM,
            cyan=Colors.CYAN,
            levelname=record.levelname,
            asctime=self.formatTime(record, "%H:%M:%S"),
            name=record.name,
            message=record.getMessage(),
        )

        return formatted


# ──────────────────────────────────────────────
# Logger Factory
# ──────────────────────────────────────────────

_loggers: dict[str, logging.Logger] = {}


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """
    Ambil atau buat logger dengan nama tertentu.

    Parameters
    ----------
    name : str
        Nama logger (biasanya ``__name__``).
    level : int
        Logging level. Default: INFO.
    log_file : str | Path, optional
        Path file log. Jika None, hanya output ke terminal.

    Returns
    -------
    logging.Logger
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Console handler (colored)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(ColoredFormatter())
        logger.addHandler(console)

    # File handler (plain)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "[%(levelname)-8s] %(asctime)s %(name)s → %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger
