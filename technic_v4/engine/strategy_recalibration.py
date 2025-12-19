"""Auto-adaptive strategy parameter retuning based on rolling Sharpe."""

from __future__ import annotations

import numpy as np
from typing import Dict, Any


def retune_parameters(sharpe_series: np.ndarray, param_config: Dict[str, Any]) -> Dict[str, Any]:
    """Adjust risk_factor based on the latest Sharpe observation.

    Args:
        sharpe_series: Array of rolling Sharpe ratios (most recent last).
        param_config: Mutable config dict containing 'risk_factor'.

    Returns:
        Updated param_config.
    """
    if sharpe_series.size == 0:
        return param_config
    latest = float(sharpe_series[-1])
    if latest < 0.5:
        param_config["risk_factor"] *= 0.9
    elif latest > 1.5:
        param_config["risk_factor"] *= 1.1
    return param_config


__all__ = ["retune_parameters"]
