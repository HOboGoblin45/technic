from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from technic_v4.engine import feature_engine
from technic_v4.data_layer.fundamentals import FundamentalsSnapshot

_DEFAULT_WEIGHTS = {
    "trend_weight": 3.0,
    "momentum_weight": 3.0,
    "volume_weight": 2.0,
    "volatility_weight": 1.0,
    "oscillator_weight": 1.0,
    "breakout_weight": 1.0,
}
_SCORING_WEIGHTS: Optional[dict] = None
_WEIGHTS_PATH = Path("technic_v4/config/scoring_weights.json")


def load_scoring_weights() -> dict:
    """
    Load subscore weights from config. Falls back to defaults if missing or invalid.
    """
    global _SCORING_WEIGHTS
    if _SCORING_WEIGHTS is not None:
        return _SCORING_WEIGHTS

    weights = _DEFAULT_WEIGHTS.copy()
    try:
        if _WEIGHTS_PATH.exists():
            with _WEIGHTS_PATH.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    weights.update({k: float(v) for k, v in loaded.items() if k in weights})
    except Exception:
        # Keep defaults on any issue
        pass

    _SCORING_WEIGHTS = weights
    return weights


def _clip(val: float, lo: float = -3, hi: float = 3) -> float:
    return max(lo, min(hi, val))


def compute_scores(
    df: pd.DataFrame,
    trade_style: str | None = None,
    fundamentals: FundamentalsSnapshot | None = None,
) -> pd.DataFrame:
    """
    Compute subscores and TechRating using centralized features (latest bar only).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    feats = feature_engine.build_features(df, fundamentals)
    if feats.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=[df.index[-1]])

    trend = 0
    if feats.get("sma_20") and feats.get("sma_50") and feats["sma_20"] > feats["sma_50"]:
        trend += 1
    if feats.get("sma_50") and feats.get("sma_200") and feats["sma_50"] > feats["sma_200"]:
        trend += 1
    if feats.get("sma_20_above_50") == 1:
        trend += 1
    trend = _clip(trend)

    momentum = 0
    rsi = feats.get("rsi_14")
    if pd.notna(rsi):
        if rsi > 55:
            momentum += 1
        if rsi > 65:
            momentum += 1
        if rsi < 45:
            momentum -= 1
        if rsi < 40:
            momentum -= 1
    if feats.get("macd_hist", 0) > 0:
        momentum += 1
    if feats.get("pct_from_high20", -999) > -3:
        momentum += 1
    momentum = _clip(momentum)

    volume_score = 0
    vsr = feats.get("vol_spike_ratio")
    if pd.notna(vsr):
        if vsr > 1.5:
            volume_score += 2
        elif vsr > 1.2:
            volume_score += 1
        elif vsr < 0.7:
            volume_score -= 1
    volume_score = _clip(volume_score)

    vol_score = 0
    atr = feats.get("atr_pct_14")
    if pd.notna(atr):
        if atr < 0.01:
            vol_score += 1
        elif atr > 0.03:
            vol_score -= 1
    vol_score = _clip(vol_score)

    osc_score = 0
    if pd.notna(rsi):
        if 55 <= rsi <= 65:
            osc_score += 1
        elif rsi > 70 or rsi < 40:
            osc_score -= 1
    osc_score = _clip(osc_score)

    breakout_score = 0
    if feats.get("pct_from_high20", -999) > -1:
        breakout_score += 1
    if feats.get("ret_5d", 0) > 0.02:
        breakout_score += 1
    breakout_score = _clip(breakout_score)

    explosiveness = max(0.0, feats.get("ret_5d", 0) or 0)

    risk_score = 1 - (feats.get("atr_pct_14", 0) or 0) * 50

    weights = load_scoring_weights()
    tech_rating = (
        weights.get("trend_weight", 0) * trend
        + weights.get("momentum_weight", 0) * momentum
        + weights.get("volume_weight", 0) * volume_score
        + weights.get("volatility_weight", 0) * vol_score
        + weights.get("oscillator_weight", 0) * osc_score
        + weights.get("breakout_weight", 0) * breakout_score
    )

    # Carry forward price fields needed by trade planner
    if "Close" in df.columns:
        out["Close"] = float(df["Close"].iloc[-1])
    if "Open" in df.columns:
        out["Open"] = float(df["Open"].iloc[-1])
    if "High" in df.columns:
        out["High"] = float(df["High"].iloc[-1])
    if "Low" in df.columns:
        out["Low"] = float(df["Low"].iloc[-1])
    if "Volume" in df.columns:
        out["Volume"] = float(df["Volume"].iloc[-1])

    # Backward-compatible ATR fields
    if "Close" in out and pd.notna(atr):
        out["ATR14_pct"] = float(atr)
        out["ATR14"] = float(atr * out["Close"])

    out["TrendScore"] = trend
    out["MomentumScore"] = momentum
    out["VolumeScore"] = volume_score
    out["VolatilityScore"] = vol_score
    out["OscillatorScore"] = osc_score
    out["BreakoutScore"] = breakout_score
    out["ExplosivenessScore"] = explosiveness
    out["RiskScore"] = risk_score
    out["TechRating"] = tech_rating
    out["AlphaScore"] = tech_rating  # placeholder
    out["TradeType"] = "None"

    return out
