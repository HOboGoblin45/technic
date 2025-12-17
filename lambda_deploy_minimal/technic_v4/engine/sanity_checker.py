"""Alternative signal sanity checker based on Sharpe ratio threshold."""

from __future__ import annotations

import numpy as np


def signal_sanity_check(signal_returns: np.ndarray, min_sharpe: float = 0.3) -> bool:
    """Return True if signal Sharpe exceeds the minimum threshold.

    Args:
        signal_returns: Array of historical signal returns.
        min_sharpe: Minimum Sharpe ratio to accept the signal.

    """
    if signal_returns.size == 0:
        return False
    mean = float(np.mean(signal_returns))
    std = float(np.std(signal_returns))
    sharpe = mean / (std + 1e-6)
    return sharpe > min_sharpe


__all__ = ["signal_sanity_check"]
