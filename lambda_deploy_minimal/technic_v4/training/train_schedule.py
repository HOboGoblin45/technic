"""Cross-validation rotation scheduler."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable


LOG_DIR = Path("models/validation_logs")


def weekly_cv_scheduler(run_cv: Callable[[], dict]) -> Path:
    """Run automated weekly temporal CV and log performance."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results = run_cv()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = LOG_DIR / f"cv_{ts}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "results": results}, f, indent=2)
    return out


__all__ = ["weekly_cv_scheduler", "LOG_DIR"]
