"""Real-time trade journal logger."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Union

LOG_FILE = Path("trade_log.json")


def log_trade(symbol: str, reason: str, confidence: Union[float, int]) -> None:
    """Append a trade entry with metadata to a JSONL file."""
    entry = {
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "reason": reason,
        "confidence": float(confidence),
    }
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        json.dump(entry, f)
        f.write("\n")


__all__ = ["log_trade"]
