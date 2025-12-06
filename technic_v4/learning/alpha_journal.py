"""Journaled alpha versioning and decay tracking."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

JOURNAL_PATH = Path("learning/alpha_journal.json")


def log_decay(model_version: str, decay_rate: float, threshold: float = 0.4) -> bool:
    """Weekly log alpha decay rate; return True if retrain suggested."""
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "decay_rate": decay_rate,
    }
    data = []
    if JOURNAL_PATH.exists():
        try:
            data = json.loads(JOURNAL_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = []
    data.append(entry)
    JOURNAL_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return decay_rate > threshold


__all__ = ["log_decay", "JOURNAL_PATH"]
