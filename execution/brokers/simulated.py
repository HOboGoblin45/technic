"""Simulated execution broker (dry run mode)."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict

LOG_PATH = Path("logs/simulated_orders.csv")


def submit_order(order: Dict) -> Dict:
    """Simulate fill and log to CSV."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fill_price = order.get("price", 0.0)
    filled = dict(order)
    filled["filled_at"] = datetime.utcnow().isoformat()
    filled["fill_price"] = fill_price
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(filled.keys()))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(filled)
    return filled


__all__ = ["submit_order", "LOG_PATH"]
