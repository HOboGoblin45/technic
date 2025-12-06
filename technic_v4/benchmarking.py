"""Continuous benchmark integration for nightly runs."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Dict


DATA_DIR = Path("data/benchmarks")


def run_benchmark_comparison(benchmarks: Dict[str, float]) -> Path:
    """Store benchmark results with timestamps for nightly runs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = DATA_DIR / f"benchmarks_{ts}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "benchmarks": benchmarks}, f, indent=2)
    return out_path


__all__ = ["run_benchmark_comparison", "DATA_DIR"]
