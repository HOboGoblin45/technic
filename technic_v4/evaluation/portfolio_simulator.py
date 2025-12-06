"""Lightweight portfolio simulator with transaction cost handling.

This module provides a simple bootstrap-style simulator to evaluate how
predictions might perform after applying per-trade execution costs. It samples
differences between predicted and actual returns with replacement to estimate
expected mean and volatility of simulated performance.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def simulate_portfolio_returns(
    preds: np.ndarray,
    actuals: np.ndarray,
    costs: float = 0.0015,
    trials: int = 1000,
) -> Tuple[float, float]:
    """Estimate portfolio return distribution accounting for execution costs.

    Args:
        preds: Predicted returns array.
        actuals: Realized returns array aligned with preds.
        costs: Per-trade cost (e.g., 15 bps = 0.0015).
        trials: Number of bootstrap resamples to run.

    Returns:
        A tuple of (mean_return, std_return) from the simulated distribution.
    """
    if preds.shape != actuals.shape:
        raise ValueError("preds and actuals must have the same shape")
    if preds.size == 0:
        return 0.0, 0.0

    diffs = preds - actuals
    returns = diffs - costs
    samples = np.random.choice(returns, size=(trials, returns.size), replace=True)
    simulated = samples.mean(axis=1)
    return float(simulated.mean()), float(simulated.std())


__all__ = ["simulate_portfolio_returns"]
