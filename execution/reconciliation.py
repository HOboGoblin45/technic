"""Trade reconciliation engine."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict

AUDIT_PATH = Path("logs/order_audit.csv")


def reconcile(submitted: List[Dict], filled: List[Dict]) -> None:
    """Compare submitted vs filled trades and log audit."""
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for sub, fil in zip(submitted, filled):
            writer.writerow([sub.get("id"), sub, fil])


__all__ = ["reconcile", "AUDIT_PATH"]
