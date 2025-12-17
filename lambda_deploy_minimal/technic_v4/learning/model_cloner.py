"""Model cloning framework."""

from __future__ import annotations

import shutil
from pathlib import Path
from datetime import datetime


def clone_live_model(live_path: Path = Path("models/active"), clone_dir: Path = Path("models/clones")) -> Path:
    """Clone live model directory with a timestamp (e.g., weekly on Sunday)."""
    ts = datetime.utcnow().strftime("%Y%m%d")
    dest = clone_dir / ts
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(live_path, dest)
    return dest


__all__ = ["clone_live_model"]
