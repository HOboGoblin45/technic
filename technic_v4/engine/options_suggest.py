from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from technic_v4 import data_engine
from technic_v4.infra.logging import get_logger

logger = get_logger()


def _days_to_exp(exp_str: str) -> Optional[int]:
    try:
        exp_dt = pd.to_datetime(exp_str)
        return (exp_dt.date() - datetime.utcnow().date()).days
    except Exception:
        return None


def suggest_option_trades(
    symbol: str,
    spot_price: float,
    bullish: bool = True,
    dte_min: int = 30,
    dte_max: int = 60,
    moneyness_tol: float = 0.07,
    min_oi: int = 200,
    min_volume: int = 20,
    max_spread_pct: float = 0.06,
) -> List[Dict[str, Any]]:
    """
    Best-effort option idea generator. Returns a small list of JSON-serializable dicts.
    """
    try:
        chain = data_engine.get_options_chain(symbol)
    except Exception:
        return []
    if chain is None or chain.empty:
        return []

    df = chain.copy()
    # Normalize columns
    type_col = "type" if "type" in df.columns else "option_type" if "option_type" in df.columns else None
    exp_col = "expiration" if "expiration" in df.columns else "expiration_date" if "expiration_date" in df.columns else None
    strike_col = "strike" if "strike" in df.columns else "strike_price" if "strike_price" in df.columns else None
    delta_col = "delta" if "delta" in df.columns else None
    iv_col = "iv" if "iv" in df.columns else "implied_volatility" if "implied_volatility" in df.columns else None
    oi_col = "open_interest" if "open_interest" in df.columns else "oi" if "oi" in df.columns else None
    bid_col = "bid_price" if "bid_price" in df.columns else "bid" if "bid" in df.columns else None
    ask_col = "ask_price" if "ask_price" in df.columns else "ask" if "ask" in df.columns else None
    mid_col = "mark" if "mark" in df.columns else "mid" if "mid" in df.columns else "last_quote" if "last_quote" in df.columns else None

    if not type_col or not exp_col or not strike_col:
        return []

    # Build mid/mark if missing
    if not mid_col and bid_col and ask_col and bid_col in df.columns and ask_col in df.columns:
        df["mid_tmp"] = (pd.to_numeric(df[bid_col], errors="coerce") + pd.to_numeric(df[ask_col], errors="coerce")) / 2
        mid_col = "mid_tmp"

    # Filter expiries
    df["days_to_exp"] = df[exp_col].apply(_days_to_exp)
    df = df[df["days_to_exp"].between(dte_min, dte_max, inclusive="both")]
    if df.empty:
        return []

    # Liquidity filter
    if oi_col and oi_col in df.columns:
        df = df[df[oi_col].fillna(0) >= min_oi]
    if "volume" in df.columns:
        df = df[df["volume"].fillna(0) >= min_volume]
    # Bid-ask spread filter
    if bid_col and ask_col and bid_col in df.columns and ask_col in df.columns:
        bid = pd.to_numeric(df[bid_col], errors="coerce")
        ask = pd.to_numeric(df[ask_col], errors="coerce")
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid.replace(0, pd.NA)
        df = df.assign(spread_pct=spread_pct)
        df = df[df["spread_pct"].fillna(0) <= max_spread_pct]
    if df.empty:
        return []

    # Directional filter
    direction = "call" if bullish else "put"
    df = df[df[type_col].str.lower() == direction]
    if df.empty:
        return []

    # Choose delta window for simple long call/put
    if delta_col and delta_col in df.columns:
        df = df[df[delta_col].between(0.35, 0.65)]

    if df.empty:
        return []

    # Near-the-money filter (moneyness within tolerance)
    try:
        moneyness = df[strike_col].astype(float) / float(spot_price)
        df = df[moneyness.sub(1.0).abs() <= moneyness_tol]
    except Exception:
        pass

    if df.empty:
        return []

    # "Sweetness" score (0-100) combining liquidity, delta, DTE, IV rank, moneyness
    df = df.copy()
    delta_abs = df[delta_col].abs() if delta_col and delta_col in df.columns else pd.Series(0.5, index=df.index)
    iv = pd.to_numeric(df[iv_col], errors="coerce") if iv_col and iv_col in df.columns else pd.Series(np.nan, index=df.index)
    oi = pd.to_numeric(df[oi_col], errors="coerce") if oi_col and oi_col in df.columns else pd.Series(np.nan, index=df.index)
    vol = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else pd.Series(np.nan, index=df.index)
    spread_rank = (1 - df.get("spread_pct", pd.Series(0, index=df.index)).rank(pct=True).fillna(0))

    iv_rank = iv.rank(pct=True).fillna(0)

    # Component scores 0-1
    delta_score = (1 - (delta_abs - 0.5).abs() / 0.25).clip(lower=0, upper=1)
    # DTE score: centered around 45 days within [dte_min, dte_max], penalize far tails
    dte = pd.to_numeric(df["days_to_exp"], errors="coerce")
    dte_score = (1 - (dte - ((dte_min + dte_max) / 2)).abs() / ((dte_max - dte_min) / 2)).clip(lower=0, upper=1)
    moneyness = pd.to_numeric(df[strike_col], errors="coerce") / float(spot_price)
    moneyness_score = (1 - (moneyness - 1.0).abs() / (moneyness_tol + 1e-9)).clip(lower=0, upper=1)
    liquidity_score = (
        oi.rank(pct=True).fillna(0) * 0.4
        + vol.rank(pct=True).fillna(0) * 0.4
        + spread_rank * 0.2
    )
    iv_score = (1 - iv_rank).clip(lower=0, upper=1)

    df["option_sweetness_score"] = (
        0.30 * liquidity_score
        + 0.25 * delta_score
        + 0.20 * dte_score
        + 0.15 * iv_score
        + 0.10 * moneyness_score
    ) * 100.0

    df = df.sort_values("option_sweetness_score") if "option_sweetness_score" in df.columns else df
    top = df.tail(3) if len(df) > 3 else df

    picks: List[Dict[str, Any]] = []
    for _, pick in top.iterrows():
        premium = float(pick.get(mid_col) or pick.get("last_trade_price") or 0.0)
        strike = float(pick[strike_col])
        expiry = str(pick[exp_col])
        delta_val = float(pick[delta_col]) if delta_col and delta_col in pick else None
        iv_val = float(pick[iv_col]) if iv_col and iv_col in pick else None
        quality_val = float(pick.get("option_sweetness_score")) if "option_sweetness_score" in pick else None
        iv_rank_val = float(iv_rank.loc[pick.name]) if pick.name in iv_rank.index else None
        iv_risk_flag = bool(iv_val and iv_val > 1.0)
        spread_val = float(pick.get("spread_pct")) if "spread_pct" in pick else None
        oi_val = float(pick.get(oi_col)) if oi_col and oi_col in pick else None
        vol_val = float(pick.get("volume")) if "volume" in pick else None
        mny_val = float(moneyness.loc[pick.name]) if pick.name in moneyness.index else None

        high_iv_flag = bool((iv_rank_val is not None and iv_rank_val >= 0.9) or (iv_val and iv_val > 1.5))

        picks.append(
            {
                "strategy_type": "long_call" if bullish else "long_put",
                "symbol": symbol,
                "strike": strike,
                "expiry": expiry,
                "delta": delta_val,
                "iv": iv_val,
                "premium": premium,
                "max_loss": premium * 100 if premium else None,
                "max_gain": None,  # uncapped
                "days_to_exp": pick.get("days_to_exp"),
                "option_sweetness_score": quality_val,
                "iv_risk_flag": iv_risk_flag,
                "iv_rank": iv_rank_val,
                "spread_pct": spread_val,
                "open_interest": oi_val,
                "volume": vol_val,
                "moneyness": mny_val,
                "high_iv_flag": high_iv_flag,
            }
        )

    # Simple vertical spread suggestion (OTM buy/sell)
    spread = None
    try:
        otm = df[df[strike_col] > spot_price] if bullish else df[df[strike_col] < spot_price]
        otm = otm.sort_values(strike_col)
        if len(otm) >= 2:
            buy_leg = otm.iloc[0]
            sell_leg = otm.iloc[1]
            debit = (buy_leg[mid_col] if mid_col in buy_leg else 0) - (sell_leg[mid_col] if mid_col in sell_leg else 0)
            spread = {
                "strategy_type": "bull_call_spread" if bullish else "bear_put_spread",
                "symbol": symbol,
                "buy_strike": float(buy_leg[strike_col]),
                "sell_strike": float(sell_leg[strike_col]),
                "expiry": str(buy_leg[exp_col]),
                "debit": float(debit) if debit is not None else None,
                "max_loss": float(debit * 100) if debit else None,
                "max_gain": float((sell_leg[strike_col] - buy_leg[strike_col] - debit) * 100) if debit else None,
                "days_to_exp": buy_leg.get("days_to_exp"),
                "iv_rank": iv_rank_val,
                "spread_pct": spread_val,
            }
    except Exception as exc:
        logger.warning("[options] spread construction failed for %s: %s", symbol, exc)

    if spread:
        picks.append(spread)

    # Aggregate OptionQualityScore at underlying level: 0.6*max + 0.4*avg top3 sweetness
    if picks:
        sweetness_vals = [p.get("option_sweetness_score") for p in picks if p.get("option_sweetness_score") is not None]
        if sweetness_vals:
            max_sw = max(sweetness_vals)
            avg_top3 = sum(sweetness_vals) / len(sweetness_vals)
            quality_agg = 0.6 * max_sw + 0.4 * avg_top3
            picks[0]["option_quality_agg"] = quality_agg
    return picks


__all__ = ["suggest_option_trades"]
