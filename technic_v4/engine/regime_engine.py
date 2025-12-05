"""
Simple regime classifier for market trend/volatility context.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _label_trend(closes: pd.Series) -> str:
    if closes is None or closes.empty:
        return "UNKNOWN"
    ma50 = closes.rolling(50).mean()
    ma200 = closes.rolling(200).mean()
    if ma50.iloc[-1] > ma200.iloc[-1] and closes.iloc[-1] > ma50.iloc[-1]:
        return "TRENDING_UP"
    if ma50.iloc[-1] < ma200.iloc[-1] and closes.iloc[-1] < ma50.iloc[-1]:
        return "TRENDING_DOWN"
    return "SIDEWAYS"


def _label_vol(returns: pd.Series) -> str:
    if returns is None or returns.empty:
        return "UNKNOWN"
    vol20 = returns.tail(20).std() * np.sqrt(252)
    vol60 = returns.tail(60).std() * np.sqrt(252)
    if pd.isna(vol20) or pd.isna(vol60) or vol60 == 0:
        return "UNKNOWN"
    ratio = vol20 / vol60
    if ratio > 1.25:
        return "HIGH_VOL"
    if ratio < 0.8:
        return "LOW_VOL"
    return "MID_VOL"


def classify_regime(spy_history: pd.DataFrame) -> Dict[str, str]:
    """
    Classify trend/volatility regime from SPY (or broad index) history.
    Expects a DataFrame with 'Close'.
    """
    closes = spy_history.get("Close")
    rets = closes.pct_change() if closes is not None else pd.Series(dtype=float)
    return {
        "trend": _label_trend(closes),
        "vol": _label_vol(rets),
    }
