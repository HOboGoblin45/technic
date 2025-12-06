"""Deployment snapshot archiver."""

from __future__ import annotations

import shutil
from pathlib import Path
from datetime import datetime


def archive_snapshot():
    ts = datetime.utcnow().strftime("%Y-%m-%d")
    out_dir = Path("backups/snapshots")
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{ts}.zip"
    targets = [
        "models/model_registry.json",
        "feedback.db",
        "training",
        "technic_v4/scanner_output",
    ]
    for t in targets:
        if Path(t).exists():
            shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=".", base_dir=t)
    return zip_path


__all__ = ["archive_snapshot"]
