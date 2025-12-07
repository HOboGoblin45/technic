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


def suggest_option_trades(symbol: str, spot_price: float, bullish: bool = True) -> List[Dict[str, Any]]:
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
    mid_col = "mark" if "mark" in df.columns else "mid" if "mid" in df.columns else "last_quote" if "last_quote" in df.columns else None

    if not type_col or not exp_col or not strike_col:
        return []

    # Filter expiries 30-90 days
    df["days_to_exp"] = df[exp_col].apply(_days_to_exp)
    df = df[df["days_to_exp"].between(30, 90, inclusive="both")]
    if df.empty:
        return []

    # Liquidity filter
    if oi_col and oi_col in df.columns:
        df = df[df[oi_col] >= 50]
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

    df = df.sort_values(oi_col) if oi_col in df.columns else df
    pick = df.iloc[-1]

    premium = float(pick.get(mid_col) or pick.get("last_trade_price") or 0.0)
    strike = float(pick[strike_col])
    expiry = str(pick[exp_col])
    delta_val = float(pick[delta_col]) if delta_col and delta_col in pick else None
    iv_val = float(pick[iv_col]) if iv_col and iv_col in pick else None

    long_call = {
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
    }

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
            }
    except Exception as exc:
        logger.warning("[options] spread construction failed for %s: %s", symbol, exc)

    picks: List[Dict[str, Any]] = [long_call]
    if spread:
        picks.append(spread)
    return picks


__all__ = ["suggest_option_trades"]
