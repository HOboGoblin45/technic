"""Strategy Stability Index (SSI) calculation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_ssi(returns_df: pd.DataFrame) -> float:
    """Quantify consistency of strategy performance across conditions."""
    volatility = returns_df.std().mean()
    return 1.0 / (1.0 + float(volatility))


__all__ = ["calculate_ssi"]
