"""Frontend error logging console placeholder."""

from __future__ import annotations

from pathlib import Path
import logging


LOG_PATH = Path("logs/frontend_error.log")
logging.basicConfig(filename=LOG_PATH, level=logging.INFO)


def log_frontend_error(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.info(message)


__all__ = ["log_frontend_error", "LOG_PATH"]
