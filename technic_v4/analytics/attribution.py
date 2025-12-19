"""Portfolio attribution engine tracking signal contributions to performance."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Dict


def write_attribution(contribs: Iterable[Dict], path: Path = Path("logs/performance_attribution.csv")) -> None:
    """Write contribution rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["symbol", "contribution", "signal", "date"])
        if f.tell() == 0:
            writer.writeheader()
        for row in contribs:
            writer.writerow(row)


__all__ = ["write_attribution"]
