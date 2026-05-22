"""Logging setup for PantryPal."""

from __future__ import annotations

import logging
import os


def configure_logging(level: int | None = None) -> None:
    """Configure a compact console logger for the application."""

    resolved_level = level
    if resolved_level is None:
        level_name = os.getenv("PANTRYPAL_LOG_LEVEL", "WARNING").upper()
        resolved_level = getattr(logging, level_name, logging.WARNING)

    logging.basicConfig(
        level=resolved_level,
        format="%(levelname)s:%(name)s:%(message)s",
    )
