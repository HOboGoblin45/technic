"""Feature drift monitor using Jensen–Shannon divergence."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon
from pathlib import Path
import csv


def compute_js_divergence(live: np.ndarray, baseline: np.ndarray) -> float:
    live_p = np.abs(live) + 1e-9
    base_p = np.abs(baseline) + 1e-9
    live_p = live_p / live_p.sum()
    base_p = base_p / base_p.sum()
    return float(jensenshannon(live_p, base_p))


def log_drift(score: float, threshold: float = 0.2, path: Path = Path("alerts/system_warnings.csv")) -> None:
    if score < threshold:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_drift", score])


__all__ = ["compute_js_divergence", "log_drift"]
