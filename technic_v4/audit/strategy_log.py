"""Live strategy audit logger."""

from __future__ import annotations

import json
import time
from pathlib import Path

LOG_PATH = Path("live_audit.json")


def log_decision(entry) -> None:
    """Append a decision/data/config entry to a JSONL audit log."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        json.dump({"ts": time.time(), "log": entry}, f)
        f.write("\n")


__all__ = ["log_decision", "LOG_PATH"]
