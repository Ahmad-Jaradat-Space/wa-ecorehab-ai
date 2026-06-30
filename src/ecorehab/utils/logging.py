"""Logging setup. Uses rich for pretty console output when available."""

from __future__ import annotations

import logging

try:  # rich is a core dep, but degrade gracefully if it is missing.
    from rich.logging import RichHandler

    _HAVE_RICH = True
except Exception:  # pragma: no cover - rich import is exercised in practice
    _HAVE_RICH = False

_CONFIGURED = False


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging once. Safe to call repeatedly."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    if _HAVE_RICH:
        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
        fmt = "%(message)s"
    else:  # pragma: no cover
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="[%X]", handlers=[handler])
    _CONFIGURED = True


def get_logger(name: str, level: int | str = logging.INFO) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: logger name, conventionally ``__name__`` of the caller.
        level: log level for the root configuration on first call.
    """
    configure_logging(level)
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
