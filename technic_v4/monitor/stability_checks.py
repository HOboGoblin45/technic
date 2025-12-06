"""System reliability audit hooks."""

from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime


def log_stability(uptime_pct: float, scan_success: float, drift_alerts: int, path: Path = Path("logs/stability_checks.csv")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.utcnow().isoformat(), uptime_pct, scan_success, drift_alerts])


__all__ = ["log_stability"]
