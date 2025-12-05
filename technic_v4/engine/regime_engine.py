"""
Regime classifier for market trend/volatility context.
Provides a simple, deterministic tagging with an extensible state_id.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _label_trend(closes: pd.Series) -> str:
    if closes is None or closes.empty:
        return "SIDEWAYS"
    ma50 = closes.rolling(50).mean()
    ma200 = closes.rolling(200).mean()
    if ma50.iloc[-1] > ma200.iloc[-1] and closes.iloc[-1] > ma50.iloc[-1]:
        return "TRENDING_UP"
    if ma50.iloc[-1] < ma200.iloc[-1] and closes.iloc[-1] < ma50.iloc[-1]:
        return "TRENDING_DOWN"
    return "SIDEWAYS"


def _label_vol(returns: pd.Series) -> str:
    if returns is None or returns.empty:
        return "LOW_VOL"
    vol20 = returns.tail(20).std() * np.sqrt(252)
    vol60 = returns.tail(60).std() * np.sqrt(252)
    if pd.isna(vol20) or pd.isna(vol60) or vol60 == 0:
        return "LOW_VOL"
    ratio = vol20 / vol60
    if ratio > 1.25:
        return "HIGH_VOL"
    if ratio < 0.8:
        return "LOW_VOL"
    return "LOW_VOL"


def classify_spy_regime(spy_history: pd.DataFrame) -> Dict[str, str | int]:
    """
    Classify trend/volatility regime from SPY (or broad index) history.
    Expects a DataFrame with 'Close'. Returns a dict with:
      - trend: TRENDING_UP / TRENDING_DOWN / SIDEWAYS
      - vol: LOW_VOL / HIGH_VOL
      - state_id: int (0..3) mapping trend/vol combinations
    """
    closes = spy_history.get("Close") if spy_history is not None else None
    rets = closes.pct_change() if closes is not None else pd.Series(dtype=float)
    trend = _label_trend(closes)
    vol = _label_vol(rets)

    state_map = {
        ("TRENDING_UP", "LOW_VOL"): 0,
        ("TRENDING_UP", "HIGH_VOL"): 1,
        ("TRENDING_DOWN", "LOW_VOL"): 2,
        ("TRENDING_DOWN", "HIGH_VOL"): 3,
        ("SIDEWAYS", "LOW_VOL"): 4,
        ("SIDEWAYS", "HIGH_VOL"): 5,
    }
    state_id = state_map.get((trend, vol), 4)
    return {"trend": trend, "vol": vol, "state_id": state_id}

# Backward-compatible alias
def classify_regime(spy_history: pd.DataFrame) -> Dict[str, str | int]:
    return classify_spy_regime(spy_history)
