# technic_v4/engine/scoring.py

from __future__ import annotations

import numpy as np
import pandas as pd

from technic_v4.indicators import calculate_indicators
from technic_v4.engine.factor_engine import compute_factor_bundle
from technic_v4.data_layer.fundamentals import FundamentalsSnapshot

def _assign_signals(df: pd.DataFrame, style: str) -> pd.DataFrame:
    """
    Assign Signal = 'Strong Long' / 'Long' / 'Avoid' based on TechRating
    and trade style.

    Option B philosophy:
      - ~8–12% of names become Strong Long (top decile-ish)
      - Next tier become Long
      - Rest are Avoid
    """
    s = (style or "").lower()

    if "TechRating" not in df.columns:
        df["Signal"] = "Avoid"
        return df

    tr = df["TechRating"].fillna(0.0)
    n = len(df)

    # Percentile-based thresholds with absolute floors
    if n >= 10:
        strong_cut = max(20.0, float(tr.quantile(0.90)))  # ~top 10%
        long_cut = max(16.0, float(tr.quantile(0.70)))    # ~top 30%
    elif n >= 5:
        strong_cut = max(20.0, float(tr.quantile(0.80)))
        long_cut = max(16.0, float(tr.quantile(0.60)))
    else:
        # small sample: fall back to static thresholds
        strong_cut = 22.0
        long_cut = 17.0

    base_strong = tr >= strong_cut
    base_long = tr >= long_cut

    # Start with base masks
    strong_mask = base_strong.copy()
    long_mask = base_long.copy()

    # --- Style-specific gating for Strong Long ---

    # Short-term swing: demand real "heat" in momentum/breakout/volume
    if "short-term" in s:
        momentum = df.get("MomentumScore", 0)
        breakout = df.get("BreakoutScore", 0)
        volume = df.get("VolumeScore", 0)

        strong_mask &= (
            (momentum + breakout >= 6) &  # e.g., 3 + 3, 2 + 4, etc.
            (volume >= 1)
        )

    # Position / longer-term: demand strong trend & quality, reasonable volatility
    elif "position" in s or "longer" in s:
        trend = df.get("TrendScore", 0)
        trendq = df.get("TrendQualityScore", 0)
        vol = df.get("VolatilityScore", 0)

        strong_mask &= (
            (trend >= 3) &
            (trendq >= 2) &
            (vol >= 0)
        )

    # Medium-term swing: just use the percentile thresholds (no extra gating)
    # (strong_mask already set)

    # Recompute Long mask so Strong Long takes priority
    long_mask = base_long & ~strong_mask

    # Default everything to Avoid
    df["Signal"] = "Avoid"
    df.loc[long_mask, "Signal"] = "Long"
    df.loc[strong_mask, "Signal"] = "Strong Long"

    return df


def compute_scores(
    df: pd.DataFrame,
    trade_style: str | None = None,
    fundamentals: FundamentalsSnapshot | None = None,
) -> pd.DataFrame:
    """
    Compute all Technic v4 subscores and composite TechRating.

    trade_style is optional and is used to slightly tilt the weighting
    between swing vs position trading.
    """
    style = (trade_style or "").lower()

    # Ensure indicators exist
    df = calculate_indicators(df).copy()

    # Factor bundle (one-per-symbol); apply as constant columns to the frame
    try:
        bundle = compute_factor_bundle(df, fundamentals)
        for k, v in bundle.factors.items():
            df[k] = v
    except Exception:
        pass

    # Default trade type; refined later
    df["TradeType"] = "None"

    # -----------------------
    # Trend Score  (-3 to +3)
    # -----------------------
    df["TrendScore"] = 0

    df.loc[
        (df["MA10"] > df["MA20"]) & (df["MA20"] > df["MA50"]),
        "TrendScore",
    ] += 1

    df.loc[df["SlopeMA20"] > 0, "TrendScore"] += 1
    df.loc[df["TrendStrength50"] > 1.0, "TrendScore"] += 1
    df.loc[df["ADX14"] > 20, "TrendScore"] += 1
    df.loc[df["SlopeMA20"] < 0, "TrendScore"] -= 1

    df["TrendScore"] = df["TrendScore"].clip(-3, 3)

    # --------------------------
    # Momentum Score (-3 to +3)
    # --------------------------
    df["MomentumScore"] = 0

    df.loc[df["RSI14"] > 55, "MomentumScore"] += 1
    df.loc[df["RSI14"] > 65, "MomentumScore"] += 1
    df.loc[df["MACD_hist"] > 0, "MomentumScore"] += 1
    df.loc[df["PctFromHigh20"] > -3, "MomentumScore"] += 1

    df.loc[df["RSI14"] < 45, "MomentumScore"] -= 1
    df.loc[df["MACD_hist"] < 0, "MomentumScore"] -= 1

    df["MomentumScore"] = df["MomentumScore"].clip(-3, 3)

    # --------------------------------
    # Explosiveness Score (-3 to +3)
    # --------------------------------
    df["ExplosivenessScore"] = 0

    df.loc[df["RVOL20"] > 1.3, "ExplosivenessScore"] += 1
    df.loc[df["ATR14_pct"] > 0.02, "ExplosivenessScore"] += 1

    bbw_ma = df["BB_width"].rolling(20).mean()
    df["BB_width_rel"] = df["BB_width"] / bbw_ma

    df.loc[df["BB_width_rel"] > 1.2, "ExplosivenessScore"] += 1
    df.loc[df["BB_pctB"].between(0.8, 1.1), "ExplosivenessScore"] += 1

    df["ExplosivenessScore"] = df["ExplosivenessScore"].clip(-3, 3)

    # --------------------------------
    # Breakout / Breakdown Score (-3 to +3)
    # --------------------------------
    df["BreakoutScore"] = 0

    close = df["Close"]
    high20 = close.rolling(20).max()
    high50 = close.rolling(50).max()
    low20 = close.rolling(20).min()
    low50 = close.rolling(50).min()

    df.loc[close >= high20, "BreakoutScore"] += 2
    df.loc[close >= high50, "BreakoutScore"] += 1
    df.loc[(close < high20) & (df["PctFromHigh20"] > -1.0), "BreakoutScore"] += 1

    df.loc[close <= low20, "BreakoutScore"] -= 2
    df.loc[close <= low50, "BreakoutScore"] -= 1

    df["BreakoutScore"] = df["BreakoutScore"].clip(-3, 3)

    # --------------------------------
    # Volume Score (-3 to +3)
    # --------------------------------
    df["VolumeScore"] = 0

    df.loc[df["RVOL20"] > 1.5, "VolumeScore"] += 2
    df.loc[(df["RVOL20"] > 1.2) & (df["RVOL20"] <= 1.5), "VolumeScore"] += 1
    df.loc[df["RVOL20"] < 0.7, "VolumeScore"] -= 1

    if "Volume" in df.columns:
        vol = df["Volume"]
        vol_ma20 = vol.rolling(20).mean()
        prev_close = df["Close"].shift()

        df.loc[
            (df["Close"] > prev_close)
            & (vol > vol_ma20)
            & (df["RVOL20"] > 1.2),
            "VolumeScore",
        ] += 1

        df.loc[
            (df["Close"] < prev_close)
            & (vol > vol_ma20)
            & (df["RVOL20"] > 1.2),
            "VolumeScore",
        ] -= 1

    df["VolumeScore"] = df["VolumeScore"].clip(-3, 3)

    # --------------------------------
    # Volatility Score (-3 to +3)
    # --------------------------------
    df["VolatilityScore"] = 0

    bb_rel = df["BB_width_rel"].fillna(1.0)

    df.loc[bb_rel < 0.8, "VolatilityScore"] += 2
    df.loc[(bb_rel >= 0.8) & (bb_rel < 0.9), "VolatilityScore"] += 1
    df.loc[df["ATR14_pct"] < 0.01, "VolatilityScore"] += 1
    df.loc[bb_rel > 1.5, "VolatilityScore"] -= 1
    df.loc[bb_rel > 2.0, "VolatilityScore"] -= 1

    df["VolatilityScore"] = df["VolatilityScore"].clip(-3, 3)

    # --------------------------------
    # Oscillator Score (-3 to +3)
    # --------------------------------
    df["OscillatorScore"] = 0

    df.loc[
        (df["MACD"] > df["MACD_signal"]) & (df["MACD_hist"] > 0),
        "OscillatorScore",
    ] += 1

    df.loc[df["MACD_hist"].diff() > 0, "OscillatorScore"] += 1
    df.loc[df["RSI14"].between(55, 65), "OscillatorScore"] += 1

    df.loc[
        (df["MACD"] < df["MACD_signal"]) & (df["MACD_hist"] < 0),
        "OscillatorScore",
    ] -= 1
    df.loc[df["RSI14"] < 40, "OscillatorScore"] -= 1

    df["OscillatorScore"] = df["OscillatorScore"].clip(-3, 3)

    # --------------------------------
    # Trend Quality Score (-3 to +3)
    # --------------------------------
    df["TrendQualityScore"] = 0

    df.loc[
        (df["MA10"] > df["MA20"]) & (df["MA20"] > df["MA50"]),
        "TrendQualityScore",
    ] += 1
    df.loc[df["ADX14"] > 25, "TrendQualityScore"] += 1
    df.loc[df["ADX14"] > df["ADX14"].shift(5), "TrendQualityScore"] += 1
    df.loc[df["ADX14"] < 15, "TrendQualityScore"] -= 1

    df["TrendQualityScore"] = df["TrendQualityScore"].clip(-3, 3)

    # -----------------------
    # Risk Score (0 to 1)
    # -----------------------
    df["RiskScore"] = 1 - (df["ATR14_pct"] * 50)
    df["RiskScore"] = df["RiskScore"].clip(0, 1)

    # =========================
    #  COMPOSITE TECH RATING
    #  (Option B - weighted; trade-style aware)
    # =========================
    weights = {
        "TrendScore": 3.0,
        "MomentumScore": 3.0,
        "BreakoutScore": 3.0,
        "ExplosivenessScore": 2.0,
        "VolumeScore": 2.0,
        "TrendQualityScore": 2.0,
        "VolatilityScore": 1.0,
        "OscillatorScore": 1.0,
    }

    # Short-term swing → emphasize momentum, volume, explosiveness
    if "short-term" in style:
        weights.update(
            {
                "MomentumScore": 4.0,
                "BreakoutScore": 4.0,
                "VolumeScore": 3.0,
                "ExplosivenessScore": 3.0,
                "TrendScore": 2.0,
                "TrendQualityScore": 1.0,
            }
        )

    # Position / longer-term → emphasize trend & trend quality, penalize volatility
    elif "position" in style or "longer" in style:
        weights.update(
            {
                "TrendScore": 4.0,
                "TrendQualityScore": 3.0,
                "VolatilityScore": 2.0,
                "ExplosivenessScore": 1.0,
            }
        )

    base_score = (
        weights["TrendScore"] * df["TrendScore"]
        + weights["MomentumScore"] * df["MomentumScore"]
        + weights["BreakoutScore"] * df["BreakoutScore"]
        + weights["ExplosivenessScore"] * df["ExplosivenessScore"]
        + weights["VolumeScore"] * df["VolumeScore"]
        + weights["TrendQualityScore"] * df["TrendQualityScore"]
        + weights["VolatilityScore"] * df["VolatilityScore"]
        + weights["OscillatorScore"] * df["OscillatorScore"]
    )

    df["TechRating_raw"] = base_score
    df["TechRating"] = (base_score * df["RiskScore"]).round(1)

    # Map TechRating + subscores → Signal, using trade-style-aware logic
    df = _assign_signals(df, style)

    # ---------- TRADE TYPE CLASSIFICATION ----------
    # Long trade types
    cond_breakout_long = (df["TrendScore"] >= 2) & (df["BreakoutScore"] >= 2)
    cond_trend_long = (df["TrendScore"] >= 2) & (df["MomentumScore"] >= 1) & (~cond_breakout_long)
    cond_pullback_long = (df["TrendScore"] >= 1) & (df["MomentumScore"] <= -1)

    # Short trade types (mirror logic on the downside)
    cond_breakout_short = (df["TrendScore"] <= -2) & (df["BreakoutScore"] <= -2)
    cond_trend_short = (df["TrendScore"] <= -2) & (df["MomentumScore"] <= -1) & (~cond_breakout_short)
    cond_pullback_short = (df["TrendScore"] <= -1) & (df["MomentumScore"] >= 1)

    df.loc[cond_breakout_long, "TradeType"] = "Breakout Long"
    df.loc[cond_trend_long, "TradeType"] = "Trend Long"
    df.loc[cond_pullback_long, "TradeType"] = "Pullback Long"

    df.loc[cond_breakout_short, "TradeType"] = "Breakout Short"
    df.loc[cond_trend_short, "TradeType"] = "Trend Short"
    df.loc[cond_pullback_short, "TradeType"] = "Pullback Short"

    # Anything that remains purely informational / untradeable
    df.loc[df["Signal"] == "Avoid", "TradeType"] = "Avoid"


    # =========================
    #  PRICE HINTS (ENTRY / STOP / TARGETS)
    # =========================
    close = df["Close"]
    atr_dollar = df["ATR14_pct"] * close
    high20 = close.rolling(20).max()
    low20 = close.rolling(20).min()
    ma20 = df["MA20"]
    swing_low10 = close.rolling(10).min()

    entry = pd.Series(np.nan, index=df.index)
    stop = pd.Series(np.nan, index=df.index)
    t1 = pd.Series(np.nan, index=df.index)
    t2 = pd.Series(np.nan, index=df.index)

    # Breakout Long
    mask_breakout = df["TradeType"] == "Breakout Long"
    entry_b = high20 * 1.002
    entry_b = entry_b.where(entry_b > 0, close)
    entry_b = np.maximum(entry_b, close * 1.001)
    entry.loc[mask_breakout] = entry_b[mask_breakout]
    stop.loc[mask_breakout] = (low20 - 0.5 * atr_dollar)[mask_breakout]
    t1.loc[mask_breakout] = entry[mask_breakout] + 1.0 * atr_dollar[mask_breakout]
    t2.loc[mask_breakout] = entry[mask_breakout] + 2.0 * atr_dollar[mask_breakout]

    # Trend Long
    mask_trend = df["TradeType"] == "Trend Long"
    entry_t = ma20.where(ma20 > 0, close)
    entry.loc[mask_trend] = entry_t[mask_trend]
    stop.loc[mask_trend] = (swing_low10 - 0.5 * atr_dollar)[mask_trend]
    t1.loc[mask_trend] = entry[mask_trend] + 1.0 * atr_dollar[mask_trend]
    t2.loc[mask_trend] = entry[mask_trend] + 2.0 * atr_dollar[mask_trend]

    # Pullback Long
    mask_pullback = df["TradeType"] == "Pullback Long"
    entry_p = ma20.where(ma20 > 0, close)
    entry.loc[mask_pullback] = entry_p[mask_pullback]
    stop.loc[mask_pullback] = (swing_low10 - 0.5 * atr_dollar)[mask_pullback]
    t1.loc[mask_pullback] = entry[mask_pullback] + 1.0 * atr_dollar[mask_pullback]
    t2.loc[mask_pullback] = entry[mask_pullback] + 2.0 * atr_dollar[mask_pullback]

    # Avoid: keep NaN for plan prices
    mask_avoid = df["Signal"] == "Avoid"
    entry.loc[mask_avoid] = np.nan
    stop.loc[mask_avoid] = np.nan
    t1.loc[mask_avoid] = np.nan
    t2.loc[mask_avoid] = np.nan

    df["EntryHint"] = entry.round(2)
    df["StopHint"] = stop.round(2)
    df["Target1"] = t1.round(2)
    df["Target2"] = t2.round(2)

    rr = (df["Target2"] - df["EntryHint"]) / (df["EntryHint"] - df["StopHint"])
    rr = rr.replace([np.inf, -np.inf], np.nan)
    df["RR_T2"] = rr.round(2)

    return df

