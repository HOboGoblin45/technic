"""
Simple backtest utilities using stored signals + price history.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from technic_v4.data_layer.price_layer import get_stock_history_df
from technic_v4.evaluation import metrics


def backtest_long_only(df_signals: pd.DataFrame, hold_days: int = 5) -> dict:
    """
    Simple long-only backtest: buy at next open, hold for hold_days, equal weight.
    """
    if df_signals is None or df_signals.empty:
        return {}
    returns: list[float] = []
    for _, row in df_signals.iterrows():
        sym = row.get("Symbol")
        if not sym:
            continue
        hist = get_stock_history_df(symbol=str(sym), days=hold_days + 2, use_intraday=False)
        if hist is None or hist.empty or "Open" not in hist or "Close" not in hist:
            continue
        entry = hist["Open"].iloc[-hold_days - 1]
        exit_px = hist["Close"].iloc[-1]
        if entry == 0:
            continue
        ret = (exit_px / entry) - 1.0
        returns.append(ret)
    if not returns:
        return {}
    rets = pd.Series(returns)
    return {
        "mean_return": rets.mean(),
        "sharpe": metrics.sharpe_ratio(rets),
    }
