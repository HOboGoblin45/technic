"""Fail-safe trigger logging."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


def log_fail_safe_trigger(event, model_id, metric):
    path = Path("fail_safe_events.csv")
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), model_id, event, metric])


__all__ = ["log_fail_safe_trigger"]
