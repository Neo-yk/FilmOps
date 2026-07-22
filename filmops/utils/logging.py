"""Centralized logging configuration for FilmOps."""

import logging

DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO, fmt: str = DEFAULT_FORMAT) -> None:
    """Configure the root logger. Idempotent."""
    root = logging.getLogger()
    if root.handlers:
        # Already configured; just adjust level if needed.
        root.setLevel(level)
        return
    logging.basicConfig(level=level, format=fmt)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
