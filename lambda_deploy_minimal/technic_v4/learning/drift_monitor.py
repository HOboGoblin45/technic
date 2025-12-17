"""Continuous label drift auditor."""

from __future__ import annotations

import numpy as np


def label_drift(actuals: np.ndarray, preds: np.ndarray, window: int = 30, threshold: float = 0.15) -> bool:
    """Trigger label drift if divergence exceeds threshold over window."""
    if len(actuals) < window or len(preds) < window:
        return False
    recent_a = actuals[-window:]
    recent_p = preds[-window:]
    # Simple divergence: mean absolute error ratio vs variance proxy
    mae = np.mean(np.abs(recent_a - recent_p))
    scale = np.std(recent_a) + 1e-6
    divergence = mae / scale
    return divergence > threshold


__all__ = ["label_drift"]
