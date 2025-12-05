"""
Relative strength helpers (sector and index comparison).

- Compute ratio lines (stock vs benchmark) over a given window.
- Provide a simple percentile of RS change for quick filtering.
"""

from __future__ import annotations

import pandas as pd
from technic_v4.data_layer.price_layer import get_stock_history_df


def rs_series(symbol: str, benchmark: str = "SPY", days: int = 200) -> pd.Series:
    """
    Return relative strength series (stock Close / benchmark Close).
    """
    stock = get_stock_history_df(symbol, days=days, use_intraday=False)
    bench = get_stock_history_df(benchmark, days=days, use_intraday=False)
    if stock is None or bench is None or stock.empty or bench.empty:
        return pd.Series(dtype=float)
    # align on dates
    df = pd.DataFrame({"stock": stock["Close"], "bench": bench["Close"]}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    rs = df["stock"] / df["bench"]
    return rs


def rs_change_percentile(symbol: str, benchmark: str = "SPY", days: int = 200, lookback: int = 40) -> float | None:
    """
    Compute percentile rank of recent RS change vs history.
    """
    rs = rs_series(symbol, benchmark, days)
    if rs.empty or len(rs) < lookback + 5:
        return None
    recent = (rs.iloc[-1] / rs.iloc[-lookback]) - 1.0
    history_changes = (rs / rs.shift(lookback) - 1.0).dropna()
    if history_changes.empty:
        return None
    pct = (history_changes < recent).mean() * 100.0
    return pct
