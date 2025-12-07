"""
Feature engineering utilities for Technic.

Single canonical builder for per-symbol features. Outputs a flat Series for
the latest bar so scoring/ML consume consistent names.
"""

from __future__ import annotations

from typing import Optional, Any, Dict

import numpy as np
import pandas as pd

try:  # talib may be unavailable; degrade gracefully
    import talib

    HAVE_TALIB = True
except Exception:  # pragma: no cover
    talib = None
    HAVE_TALIB = False

from technic_v4.data_layer.fundamentals import FundamentalsSnapshot
from technic_v4.config.settings import get_settings
from technic_v4.alpha import tft_inference


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] if col in df.columns else pd.Series(dtype=float)


def build_features(
    history_df: pd.DataFrame,
    fundamentals: Optional[FundamentalsSnapshot] = None,
) -> pd.Series:
    """
    Build all per-symbol features for the latest bar.
    Returns: pd.Series (snake_case feature names).
    """
    if history_df is None or history_df.empty:
        return pd.Series(dtype=float)

    df = history_df.copy()
    df = df.sort_index()
    close = _safe_series(df, "Close")
    high = _safe_series(df, "High")
    low = _safe_series(df, "Low")
    volume = _safe_series(df, "Volume")
    open_ = _safe_series(df, "Open")

    feats: Dict[str, Any] = {}

    # Returns / momentum
    feats["ret_1d"] = float(close.pct_change(1).iloc[-1]) if len(close) > 1 else np.nan
    feats["ret_5d"] = float(close.pct_change(5).iloc[-1]) if len(close) > 5 else np.nan
    feats["ret_21d"] = float(close.pct_change(21).iloc[-1]) if len(close) > 21 else np.nan

    # Realized volatility 20d
    if len(close) >= 20:
        feats["vol_realized_20"] = float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252))
    else:
        feats["vol_realized_20"] = np.nan

    # ATR 14 pct
    if HAVE_TALIB and len(close) >= 14:
        atr14 = talib.ATR(high.values, low.values, close.values, timeperiod=14)
        feats["atr_pct_14"] = float((atr14 / close.replace(0, np.nan)).iloc[-1])
    else:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        atr14 = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
        feats["atr_pct_14"] = float((atr14 / close.replace(0, np.nan)).iloc[-1]) if len(atr14) else np.nan

    # Moving averages & trend
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    feats["sma_20"] = float(ma20.iloc[-1]) if len(ma20) else np.nan
    feats["sma_50"] = float(ma50.iloc[-1]) if len(ma50) else np.nan
    feats["sma_200"] = float(ma200.iloc[-1]) if len(ma200) else np.nan
    feats["sma_20_above_50"] = float((ma20.iloc[-1] > ma50.iloc[-1]) if len(ma20) and len(ma50) else np.nan)
    feats["pct_from_high20"] = (
        float((close.iloc[-1] / close.rolling(20).max().iloc[-1] - 1) * 100) if len(close) >= 20 else np.nan
    )

    # RSI 14
    if HAVE_TALIB and len(close) >= 14:
        feats["rsi_14"] = float(talib.RSI(close.values, timeperiod=14)[-1])
    else:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        feats["rsi_14"] = float((100 - (100 / (1 + rs))).iloc[-1]) if len(rs) else np.nan

    # MACD histogram
    if HAVE_TALIB and len(close) >= 26:
        _, _, macd_hist = talib.MACD(close.values, fastperiod=12, slowperiod=26, signalperiod=9)
        feats["macd_hist"] = float(macd_hist[-1])
    else:
        macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        macd_signal = macd.ewm(span=9).mean()
        macd_hist = macd - macd_signal
        feats["macd_hist"] = float(macd_hist.iloc[-1]) if len(macd_hist) else np.nan

    # Dollar volume + spike
    dollar_vol = close * volume
    if len(dollar_vol) >= 20:
        dv20 = dollar_vol.rolling(20).mean()
        feats["dollar_vol_20"] = float(dv20.iloc[-1])
        feats["vol_spike_ratio"] = float((dollar_vol.iloc[-1] / dv20.iloc[-1]) if dv20.iloc[-1] else np.nan)
    else:
        feats["dollar_vol_20"] = np.nan
        feats["vol_spike_ratio"] = np.nan

    # Gap
    gap = (open_ - close.shift(1)).abs() / close.shift(1)
    feats["gap_mean_10"] = float(gap.rolling(10).mean().iloc[-1]) if len(gap) >= 10 else np.nan

    # Fundamentals (best-effort)
    raw_f = fundamentals.raw if fundamentals is not None else {}
    feats["value_ep"] = raw_f.get("earnings_yield") or raw_f.get("ep")
    feats["quality_roe"] = raw_f.get("return_on_equity") or raw_f.get("roe")

    # Optional TFT multi-horizon forecasts
    settings = get_settings()
    if getattr(settings, "use_tft_features", False):
        try:
            tft_feats = tft_inference.tft_predict_horizons(symbol=raw_f.get("symbol", ""), history_df=df)
            feats.update(tft_feats or {})
        except Exception:
            # best-effort; ignore TFT errors
            pass

    return pd.Series(feats)


def get_latest_features(
    history_df: pd.DataFrame,
    fundamentals: Optional[FundamentalsSnapshot] = None,
) -> pd.Series:
    """Convenience wrapper (same as build_features)."""
    return build_features(history_df, fundamentals)


def merge_tft_features(results_df: pd.DataFrame, tft_features: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join TFT forecast features onto scan results by Symbol.
    """
    if results_df is None or results_df.empty or tft_features is None or tft_features.empty:
        return results_df
    feats = tft_features.copy()
    if "Symbol" in feats.columns:
        feats = feats.set_index("Symbol")
    merged = results_df.merge(feats, left_on="Symbol", right_index=True, how="left")
    return merged


__all__ = ["build_features", "get_latest_features", "merge_tft_features"]
