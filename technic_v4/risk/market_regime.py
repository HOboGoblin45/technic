"""Market regime tagging system."""

from __future__ import annotations

import pandas as pd


def classify_market_regime(returns: pd.Series) -> str:
    """Classify current market as trending, mean-reverting, high-vol, etc."""
    volatility = returns.std()
    trend = returns.mean()
    if volatility > 0.02 and trend > 0:
        return "High Vol Trending"
    if volatility < 0.01:
        return "Low Vol"
    return "Choppy"


__all__ = ["classify_market_regime"]
