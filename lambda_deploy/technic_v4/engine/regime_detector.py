"""Dynamic risk regime detection utility."""

from __future__ import annotations

import pandas as pd


def detect_regime(vix_series: pd.Series) -> str:
    """Detect high/low volatility regimes based on VIX level."""
    if vix_series.empty:
        return "unknown"
    return "high" if float(vix_series.iloc[-1]) > 25 else "low"


__all__ = ["detect_regime"]
