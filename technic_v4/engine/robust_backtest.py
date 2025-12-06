"""Backtest robustness simulator utilities."""

from __future__ import annotations

import numpy as np


def simulate_drift(returns):
    """Add random noise to returns to stress-test robustness."""
    noise = np.random.normal(0, 0.05, len(returns))
    return returns + noise


__all__ = ["simulate_drift"]
