"""Adaptive live signal engine for threshold updates."""

from __future__ import annotations

import pandas as pd


def update_threshold(signal_df: pd.DataFrame, current_volatility: float) -> pd.DataFrame:
    """Adjust signal thresholds based on current volatility."""
    return signal_df * (1 + current_volatility * 0.1)


__all__ = ["update_threshold"]
