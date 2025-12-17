"""Weekly expected move estimator using ATM straddle prices."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime


def log_expected_move(implied_move: float, path: Path = Path("logs/implied_move.json")):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.utcnow().isoformat(), "implied_move": implied_move}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


__all__ = ["log_expected_move"]
