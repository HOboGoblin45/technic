"""Volatility-aware trade filter.

Filters out symbols or periods with short-term volatility above a threshold.
"""

from __future__ import annotations

import pandas as pd


def volatility_filter(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Filter rows where 5-day realized volatility is below a threshold.

    Args:
        df: DataFrame with a 'close' column.
        threshold: Maximum allowed rolling volatility (std of returns).

    Returns:
        Filtered DataFrame containing only rows under the volatility cap.
    """
    if "close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'close' column.")

    out = df.copy()
    out["volatility"] = out["close"].pct_change().rolling(5).std()
    return out[out["volatility"] < threshold]


__all__ = ["volatility_filter"]
