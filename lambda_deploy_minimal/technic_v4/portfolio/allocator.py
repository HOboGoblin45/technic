"""Smart allocation balancer for risk/vol/confidence-aware sizing."""

from __future__ import annotations

import numpy as np
from typing import Sequence


def smart_allocate(signals: Sequence[dict]):
    """Rebalance trades across risk tiers, volatility, and confidence scores."""
    if not signals:
        return np.array([])
    weights = np.array([s["confidence"] / s["volatility"] for s in signals])
    denom = weights.sum()
    if denom == 0:
        return np.zeros_like(weights)
    normalized = weights / denom
    return normalized


__all__ = ["smart_allocate"]
