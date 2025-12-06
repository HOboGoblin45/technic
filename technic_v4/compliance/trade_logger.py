"""Live trade compliance logger."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

LOG_PATH = Path("compliance/live_trade_log.json")


def log_trade(timestamp: datetime, symbol: str, model_id: str, user_id: str, reason: str):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": timestamp.isoformat(),
        "symbol": symbol,
        "model_id": model_id,
        "user_id": user_id,
        "reason": reason,
    }
    data = []
    if LOG_PATH.exists():
        try:
            data = json.loads(LOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = []
    data.append(entry)
    LOG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


__all__ = ["log_trade", "LOG_PATH"]
