from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

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

    # Base risk score from ATR% (roughly 1 - ATR*50)
    risk_score = 1 - (feats.get("atr_pct_14", 0) or 0) * 50

    # --- TechRating v3: risk-aware technical composite -------------------
    weights = load_scoring_weights()
    tech_raw = (
        weights.get("trend_weight", 0) * trend
        + weights.get("momentum_weight", 0) * momentum
        + weights.get("volume_weight", 0) * volume_score
        + weights.get("volatility_weight", 0) * vol_score
        + weights.get("oscillator_weight", 0) * osc_score
        + weights.get("breakout_weight", 0) * breakout_score
    )

    # risk_score can drift outside [0,1] for very high ATR; clamp it
    risk_factor = float(np.clip(risk_score, 0.0, 1.0))

    # Smooth risk scaling:
    # - Very low volatility (risk_factor ~ 1) → scale ~ 1.0
    # - Very high volatility (risk_factor ~ 0) → scale ~ 0.7
    risk_scale = 0.7 + 0.3 * risk_factor
    tech_rating = tech_raw * risk_scale

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
        atr_series = atr * out["Close"]
        out["ATR14"] = atr_series.iloc[-1]

    out["TrendScore"] = trend
    out["MomentumScore"] = momentum
    out["VolumeScore"] = volume_score
    out["VolatilityScore"] = vol_score
    out["OscillatorScore"] = osc_score
    out["BreakoutScore"] = breakout_score
    out["ExplosivenessScore"] = explosiveness
    out["RiskScore"] = risk_score
    out["risk_score"] = risk_score
    out["IsUltraRisky"] = bool(risk_score < 0.12)

    # Expose both raw and risk-adjusted TechRating for downstream engines
    out["TechRating_raw"] = tech_raw
    out["TechRating"] = tech_rating

    # For now, AlphaScore starts as TechRating; later overridden by ML alpha
    out["AlphaScore"] = tech_rating
    out["TradeType"] = "None"

    return out


def build_institutional_core_score(df: pd.DataFrame) -> pd.Series:
    """
    Build a single Institutional Core Score (ICS) in [0, 100] for each row.

    ICS blends:
      - technical quality (TechRating)
      - factor + ML alpha (alpha_blend / AlphaScore / factor_alpha)
      - stability (risk_score / RiskScore)
      - liquidity (DollarVolume)
      - fundamental quality (QualityScore or related columns)
      - event term (earnings / dividend flags)

    All inputs are best-effort: if a column is missing, that term falls
    back to a neutral 50/100.
    """
    if df.empty:
        return pd.Series([], index=df.index, dtype=float)

    idx = df.index

    def _pct_rank(series: pd.Series | None) -> pd.Series:
        if series is None or series.empty or series.isna().all():
            return pd.Series(50.0, index=idx)
        return series.rank(pct=True).astype(float) * 100.0

    # --- Tech term ---
    tech_src = None
    for col in ("TechRating", "TechRating_raw"):
        if col in df.columns:
            tech_src = df[col]
            break
    tech_term = _pct_rank(tech_src)

    # --- Alpha term (factor + ML blend) ---
    alpha_src = None
    for col in ("alpha_blend", "AlphaScorePct", "AlphaScore", "ml_alpha_z", "factor_alpha"):
        if col in df.columns and not df[col].isna().all():
            alpha_src = df[col]
            break
    alpha_term = _pct_rank(alpha_src)

    # --- Stability term (higher is more stable) ---
    risk_src = None
    for col in ("risk_score", "RiskScore"):
        if col in df.columns:
            risk_src = df[col]
            break

    if risk_src is None or risk_src.isna().all():
        stability_term = pd.Series(50.0, index=idx)
    else:
        r = risk_src.clip(lower=0.0, upper=1.0)
        stability_term = (1.0 - r) * 100.0

    # --- Liquidity term (DollarVolume) ---
    if "DollarVolume" in df.columns:
        dv = df["DollarVolume"].clip(lower=1.0)
        liquidity_term = np.log10(dv).rank(pct=True).astype(float) * 100.0
    else:
        liquidity_term = pd.Series(50.0, index=idx)

    # --- Fundamental quality term ---
    quality_src = None
    for col in ("QualityScore", "fundamental_quality_score", "quality_roe_sector_z"):
        if col in df.columns and not df[col].isna().all():
            quality_src = df[col]
            break
    quality_term = _pct_rank(quality_src)

    # Blend in sponsorship if available
    if "SponsorshipScore" in df.columns and not df["SponsorshipScore"].isna().all():
        sponsor_term = _pct_rank(df["SponsorshipScore"])
        quality_term = (quality_term + sponsor_term) / 2.0

    # --- Event term (earnings / dividends) ---
    event_term = pd.Series(50.0, index=idx)

    if "is_pre_earnings_window" in df.columns:
        event_term = event_term - 10.0 * df["is_pre_earnings_window"].fillna(False).astype(float)

    if "is_post_earnings_positive_window" in df.columns:
        event_term = event_term + 5.0 * df["is_post_earnings_positive_window"].fillna(False).astype(float)

    if "has_dividend_ex_soon" in df.columns:
        event_term = event_term + 2.0 * df["has_dividend_ex_soon"].fillna(False).astype(float)

    if "surprise_streak" in df.columns:
        streak = pd.to_numeric(df["surprise_streak"], errors="coerce").fillna(0.0)
        streak = streak.clip(lower=-5, upper=5) / 5.0  # -1 to 1
        event_term = event_term + 4.0 * streak

    if "avg_surprise_bp" in df.columns:
        avg_bp = pd.to_numeric(df["avg_surprise_bp"], errors="coerce")
        boost = (avg_bp.clip(lower=-500, upper=500) / 500.0) * 3.0  # cap at +/-500bp
        event_term = event_term + boost.fillna(0.0)

    if "dividend_yield" in df.columns:
        dy = df["dividend_yield"].clip(lower=0.0)
        sweet = dy.between(0.01, 0.06)
        very_high = dy > 0.08
        event_term = event_term + 3.0 * sweet.astype(float) - 5.0 * very_high.astype(float)

    # --- Weighted blend ---
    ics = (
        0.28 * tech_term
        + 0.22 * alpha_term
        + 0.18 * quality_term
        + 0.12 * stability_term
        + 0.10 * liquidity_term
        + 0.10 * event_term
    )

    ics = ics.clip(lower=0.0, upper=100.0)
    ics = ics.fillna(ics.median())
    return ics


__all__ = ["compute_scores", "build_institutional_core_score"]
