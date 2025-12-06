"""Live drawdown monitor."""

from __future__ import annotations

import pandas as pd


def calculate_drawdown(equity_curve: pd.Series) -> float:
    """Compute rolling max drawdown."""
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return float(drawdown.min())


__all__ = ["calculate_drawdown"]
