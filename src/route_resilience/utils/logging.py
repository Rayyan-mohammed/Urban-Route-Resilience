"""Project-wide logging built on `rich` for readable, colorized output.

One configured logger per module via `get_logger(__name__)`. Centralizing this
avoids every script calling `print()` and gives us timestamped, level-tagged
logs that look professional in demos and CI alike.
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False


def _configure_root(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    _CONFIGURED = True


def get_logger(name: str = "route_resilience", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger. Safe to call from any module."""
    _configure_root(level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
