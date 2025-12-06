"""Maintenance tools CLI."""

from __future__ import annotations

import shutil
from pathlib import Path


LOG_DIR = Path("logs/maintenance")


def purge_stale_models():
    shutil.rmtree("models/stale", ignore_errors=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "purge.log").write_text("Purged stale models\n", encoding="utf-8")


def rebuild_cache():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "cache.log").write_text("Cache rebuild triggered\n", encoding="utf-8")


def restart_ingestion():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "ingest.log").write_text("Ingestion restart requested\n", encoding="utf-8")


__all__ = ["purge_stale_models", "rebuild_cache", "restart_ingestion"]
