"""Post-processing utilities for prediction error envelopes."""

from __future__ import annotations

import numpy as np
from typing import Dict, Sequence


def generate_confidence_band(predictions: Sequence[float], residuals: Sequence[float]) -> Dict[str, float]:
    """
    Calculate bootstrapped 80% and 95% return ranges from recent live prediction errors.
    """
    preds = np.array(predictions, dtype=float)
    res = np.array(residuals, dtype=float)
    if res.size == 0:
        return {"p80_low": np.nan, "p80_high": np.nan, "p95_low": np.nan, "p95_high": np.nan}
    boot = np.random.choice(res, size=(1000, res.size), replace=True).mean(axis=1)
    p80_low, p80_high = np.percentile(boot, [10, 90])
    p95_low, p95_high = np.percentile(boot, [2.5, 97.5])
    return {
        "p80_low": float(preds.mean() + p80_low),
        "p80_high": float(preds.mean() + p80_high),
        "p95_low": float(preds.mean() + p95_low),
        "p95_high": float(preds.mean() + p95_high),
    }


__all__ = ["generate_confidence_band"]
