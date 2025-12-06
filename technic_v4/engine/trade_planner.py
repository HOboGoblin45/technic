# trade_planner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Dict, Any
import pandas as pd
import numpy as np


SignalType = Literal["Strong Long", "Long", "Strong Short", "Short", "Avoid"]


@dataclass
class RiskSettings:
    account_size: float        # e.g. 10000
    risk_pct: float            # e.g. 1.0 (percent of account per trade)
    target_rr: float           # e.g. 2.0
    trade_style: str = "swing" # placeholder for future logic
    allow_shorts: bool = False # engine can detect shorts; execute only if True
    liquidity_cap_pct: float = 5.0  # cap position to % of ADV (dollar volume)


def _dollar_atr(row: pd.Series) -> float:
    """
    Convert ATR% to absolute dollars.

    Assumes ATR14_pct is a *fraction* of price (0.02 = 2%), which matches
    the indicator + scoring logic.
    """
    close = float(row["Close"])
    atr_pct = float(row.get("ATR14_pct", 0.0) or 0.0)
    return close * atr_pct


def _infer_signal(row: pd.Series) -> SignalType:
    """Fallback if scoring did not populate Signal."""
    rating = float(row.get("TechRating", 0.0) or 0.0)
    trend = float(row.get("TrendScore", 0.0) or 0.0)
    momo  = float(row.get("MomentumScore", 0.0) or 0.0)

    if rating >= 22 and trend >= 2 and momo >= 1:
        return "Strong Long"
    if rating >= 16 and trend >= 1:
        return "Long"
    if rating <= -22 and trend <= -2 and momo <= -1:
        return "Strong Short"
    if rating <= -16 and trend <= -1:
        return "Short"
    return "Avoid"


def _base_trade_plan_dict(signal: str = "Avoid") -> Dict[str, Any]:
    return {
        "Signal": signal,
        "EntryPrice": np.nan,
        "StopPrice": np.nan,
        "TargetPrice": np.nan,
        "RewardRisk": np.nan,
        "PositionSize": 0,
        "LiquidityLimited": False,
        "LiquidityNote": None,
    }


def plan_trade_for_row(row: pd.Series, risk: RiskSettings) -> Dict[str, Any]:
    """
    Compute entry/stop/target/size for a single symbol row.
    Returns a dict you can merge back into your scan DataFrame.
    """

    signal: SignalType = row.get("Signal") or _infer_signal(row)
    trade_type = str(row.get("TradeType", "None") or "None")

    close = float(row["Close"])
    high  = float(row.get("High", close))
    low   = float(row.get("Low", close))

    atr_dollar = _dollar_atr(row)
    if atr_dollar <= 0 or close <= 0:
        return _base_trade_plan_dict("Avoid")

    # Structure references (fallbacks keep this robust)
    swing_low  = float(row.get("SwingLow10", low))
    swing_high = float(row.get("SwingHigh10", high))
    recent_low = float(row.get("RecentLow5", swing_low))
    recent_high = float(row.get("RecentHigh5", swing_high))
    ma20 = float(row.get("MA20", close))

    breakout_score = float(row.get("BreakoutScore", 0.0) or 0.0)

    # Determine side: long or short
    is_long_signal = signal in ("Strong Long", "Long")
    is_short_signal = signal in ("Strong Short", "Short")

    # Long-only UI: if shorts are not allowed, never execute short trades
    if is_short_signal and not risk.allow_shorts:
        # Keep the Signal label for information, but no trade plan
        return _base_trade_plan_dict(signal=signal)

    entry = stop = target = np.nan
    rr = np.nan

    # ---------- LONG SIDE ----------
    if is_long_signal:
        # Stop placement: structural + ATR guard rails
        raw_stop_candidates = [
            swing_low,
            recent_low,
            close - 1.5 * atr_dollar,
        ]
        stop = min(raw_stop_candidates)

        # Enforce stop sanity: at least 0.5 ATR away, at most 3 ATR away
        max_stop = close - 0.5 * atr_dollar
        min_stop = close - 3.0 * atr_dollar
        stop = max(stop, min_stop)
        stop = min(stop, max_stop)

        if stop >= close:
            return _base_trade_plan_dict("Avoid")

        # Entry depends on TradeType
        if trade_type == "Breakout Long":
            # Enter slightly above recent resistance
            base_level = max(swing_high, recent_high, close)
            entry = base_level + 0.2 * atr_dollar

        elif trade_type == "Trend Long":
            # Enter near MA20 unless price is extended
            base_level = ma20 if ma20 > 0 else close
            # If price is very extended above MA20, wait for small pullback
            if close > base_level + 2.0 * atr_dollar:
                base_level = close - 0.5 * atr_dollar
            entry = max(base_level, close - 0.25 * atr_dollar) + 0.1 * atr_dollar

        elif trade_type == "Pullback Long":
            # Enter on recovery from pullback, slightly above close
            base_level = max(ma20, recent_low, close)
            entry = base_level + 0.05 * atr_dollar

        else:
            # Generic long: small improvement above close
            entry = close + 0.1 * atr_dollar

        risk_per_share = entry - stop
        if risk_per_share <= 0:
            return _base_trade_plan_dict("Avoid")

        target = entry + risk.target_rr * risk_per_share
        rr = (target - entry) / risk_per_share

    # ---------- SHORT SIDE (BACKEND ONLY, optional execution) ----------
    elif is_short_signal and risk.allow_shorts:
        # Mirror logic for shorts
        raw_stop_candidates = [
            swing_high,
            recent_high,
            close + 1.5 * atr_dollar,
        ]
        stop = max(raw_stop_candidates)

        min_stop = close + 0.5 * atr_dollar
        max_stop = close + 3.0 * atr_dollar
        stop = min(stop, max_stop)
        stop = max(stop, min_stop)

        if stop <= close:
            return _base_trade_plan_dict("Avoid")

        if trade_type == "Breakout Short":
            base_level = min(swing_low, recent_low, close)
            entry = base_level - 0.2 * atr_dollar
        elif trade_type == "Trend Short":
            base_level = ma20 if ma20 > 0 else close
            if close < base_level - 2.0 * atr_dollar:
                base_level = close + 0.5 * atr_dollar
            entry = min(base_level, close + 0.25 * atr_dollar) - 0.1 * atr_dollar
        elif trade_type == "Pullback Short":
            base_level = min(ma20, recent_high, close)
            entry = base_level - 0.05 * atr_dollar
        else:
            entry = close - 0.1 * atr_dollar

        risk_per_share = stop - entry
        if risk_per_share <= 0:
            return _base_trade_plan_dict("Avoid")

        target = entry - risk.target_rr * risk_per_share
        rr = (entry - target) / risk_per_share

    else:
        # Signal is "Avoid" or unknown
        return _base_trade_plan_dict("Avoid")

    # Position sizing
    dollar_risk = risk.account_size * (risk.risk_pct / 100.0)
    risk_per_share = abs(entry - stop)

    if risk_per_share <= 0 or dollar_risk <= 0:
        return _base_trade_plan_dict("Avoid")

    position_size = int(dollar_risk // risk_per_share)
    if position_size <= 0:
        # Can't trade under current risk rules
        return {
            "Signal": signal,
            "EntryPrice": round(entry, 2),
            "StopPrice": round(stop, 2),
            "TargetPrice": round(target, 2),
            "RewardRisk": round(rr, 2) if np.isfinite(rr) else np.nan,
            "PositionSize": 0,
        }

    # Liquidity cap by ADV / dollar volume
    liquidity_limited = False
    liquidity_note = None
    try:
        adv_dollar = None
        for key in ["DollarVolume20", "dollar_vol_20", "DollarVolume"]:
            if key in row and pd.notna(row.get(key)):
                adv_dollar = float(row.get(key))
                break
        if adv_dollar and adv_dollar > 0 and risk.liquidity_cap_pct > 0:
            max_dollar = (risk.liquidity_cap_pct / 100.0) * adv_dollar
            planned_value = position_size * entry
            if planned_value > max_dollar:
                capped_size = int(max_dollar // entry)
                if capped_size < position_size:
                    position_size = max(capped_size, 0)
                    liquidity_limited = True
                    liquidity_note = f"Capped to {risk.liquidity_cap_pct:.1f}% ADV (~${max_dollar:,.0f})"
    except Exception:
        pass

    return {
        "Signal": signal,
        "EntryPrice": round(entry, 2),
        "StopPrice": round(stop, 2),
        "TargetPrice": round(target, 2),
        "RewardRisk": round(rr, 2) if np.isfinite(rr) else np.nan,
        "PositionSize": position_size,
        "LiquidityLimited": liquidity_limited,
        "LiquidityNote": liquidity_note,
    }


def plan_trades(df: pd.DataFrame, risk: RiskSettings) -> pd.DataFrame:
    """Vectorized wrapper: apply plan_trade_for_row across the scan DataFrame."""
    plans = df.apply(lambda row: plan_trade_for_row(row, risk), axis=1, result_type="expand")
    for col in plans.columns:
        df[col] = plans[col]
    return df
