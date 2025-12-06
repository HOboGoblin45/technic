"""Alpha composition pipeline with entropy-based dynamic orchestration."""

from __future__ import annotations

import numpy as np
from typing import Sequence


def entropy_weight_blend(signals: Sequence[float]) -> np.ndarray:
    """Compute weights inversely proportional to signal entropy (stable predictors get higher weight)."""
    arr = np.array(signals, dtype=float)
    if arr.size == 0:
        return arr
    # Normalize absolute signals to pseudo-probabilities
    probs = np.abs(arr) + 1e-9
    probs = probs / probs.sum()
    # Entropy per predictor approximated via its contribution
    entropy = -probs * np.log(probs + 1e-12)
    inv_entropy = 1.0 / (entropy + 1e-6)
    weights = inv_entropy / inv_entropy.sum()
    return weights


def dynamic_blend(signals: Sequence[float]) -> float:
    """Blend signals using entropy weights."""
    if not signals:
        return 0.0
    w = entropy_weight_blend(signals)
    return float(np.dot(signals, w))


__all__ = ["entropy_weight_blend", "dynamic_blend"]
