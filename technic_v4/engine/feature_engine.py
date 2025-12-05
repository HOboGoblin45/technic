"""
Feature engineering utilities for Technic.

Responsibilities:
- Transform raw price history (OHLCV) + optional fundamentals snapshot into
  the feature set used by TechRating/score computation.
- Centralize indicator/factor construction so it can be reused by ML models.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
try:  # talib may be unavailable in some environments; degrade gracefully
    import talib

    HAVE_TALIB = True
except Exception:  # pragma: no cover
    talib = None
    HAVE_TALIB = False

from technic_v4.data_layer.fundamentals import FundamentalsSnapshot


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] if col in df.columns else pd.Series(dtype=float)


def build_features(
    history_df: pd.DataFrame,
    fundamentals: Optional[FundamentalsSnapshot] = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix from raw OHLCV history and optional fundamentals.

    Returns a DataFrame aligned to history_df index with:
    - Trend / momentum: returns over multiple windows, MACD, ADX, MA slopes
    - Volatility / liquidity: ATR%, realized vol, dollar volume, gap stats
    - Quality/value stubs (best-effort) based on fundamentals snapshot
    """
    if history_df is None or history_df.empty:
        return pd.DataFrame()

    df = history_df.copy()
    close = _safe_series(df, "Close")
    high = _safe_series(df, "High")
    low = _safe_series(df, "Low")
    volume = _safe_series(df, "Volume")

    feats = pd.DataFrame(index=df.index)

    # Returns / momentum
    feats["Ret_5"] = close.pct_change(5)
    feats["Ret_21"] = close.pct_change(21)
    feats["Ret_63"] = close.pct_change(63)
    feats["MomentumScore"] = feats["Ret_21"] + feats["Ret_63"]
    feats["Reversal_5"] = -feats["Ret_5"]

    # MACD signal
    if HAVE_TALIB:
        macd, macd_signal, macd_hist = talib.MACD(close.values, fastperiod=12, slowperiod=26, signalperiod=9)
    else:
        macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        macd_signal = macd.ewm(span=9).mean()
        macd_hist = macd - macd_signal
    feats["MACD"] = macd
    feats["MACD_signal"] = macd_signal
    feats["MACD_hist"] = macd_hist

    # ADX trend strength
    if HAVE_TALIB:
        feats["ADX14"] = talib.ADX(high.values, low.values, close.values, timeperiod=14)
    else:
        feats["ADX14"] = pd.Series(np.nan, index=df.index)

    # Moving averages and slopes
    if HAVE_TALIB:
        ma10 = talib.SMA(close.values, timeperiod=10)
        ma20 = talib.SMA(close.values, timeperiod=20)
        ma50 = talib.SMA(close.values, timeperiod=50)
    else:
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
    feats["MA10"] = ma10
    feats["MA20"] = ma20
    feats["MA50"] = ma50
    feats["SlopeMA20"] = pd.Series(ma20, index=df.index).diff()
    feats["TrendStrength50"] = pd.Series(ma50, index=df.index).pct_change()

    # ATR% and realized volatility
    if HAVE_TALIB:
        atr14 = talib.ATR(high.values, low.values, close.values, timeperiod=14)
    else:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        atr14 = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    feats["ATR14"] = atr14
    feats["ATR_pct"] = atr14 / close.replace(0, np.nan)
    feats["VolatilityScore"] = close.pct_change().rolling(20).std() * np.sqrt(252)

    # Volume / dollar volume
    feats["DollarVolume20"] = (close * volume).rolling(20).mean()
    feats["VolumeScore"] = volume.rolling(20).mean()

    # Gap stats
    open_ = _safe_series(df, "Open")
    gap = (open_ - close.shift(1)).abs() / close.shift(1)
    feats["GapStat10"] = gap.rolling(10).mean()

    # Breakout/explosiveness proxies
    feats["BreakoutScore"] = (close > ma20).astype(float) + (close > ma50).astype(float)
    feats["ExplosivenessScore"] = (feats["Ret_5"].fillna(0)).clip(lower=0)

    # Oscillator proxy (RSI)
    if HAVE_TALIB:
        feats["RSI14"] = talib.RSI(close.values, timeperiod=14)
    else:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        feats["RSI14"] = 100 - (100 / (1 + rs))
    feats["OscillatorScore"] = (feats["RSI14"] - 50.0) / 10.0

    # Quality/value placeholders from fundamentals (best-effort)
    raw_f = fundamentals.raw if fundamentals is not None else {}
    feats["value_ep"] = raw_f.get("earnings_yield") or raw_f.get("ep")
    feats["quality_roe"] = raw_f.get("return_on_equity") or raw_f.get("roe")

    return feats


def get_latest_features(
    history_df: pd.DataFrame,
    fundamentals: Optional[FundamentalsSnapshot] = None,
) -> pd.Series:
    """
    Convenience helper: build full feature set and return the latest row.
    """
    feats = build_features(history_df, fundamentals)
    if feats is None or feats.empty:
        return pd.Series(dtype=float)
    return feats.iloc[-1]
