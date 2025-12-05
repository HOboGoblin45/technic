from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _strategy_bands(trade_style: str, signal: Optional[str], tech_rating: Optional[float]) -> Tuple[Tuple[int, int], Tuple[float, float]]:
    """
    Map a trade style + conviction into (dte_range, delta_range).
    """
    style = (trade_style or "").lower()
    sig = (signal or "").lower()
    tr = tech_rating if tech_rating is not None else 0.0

    # Base bands by style
    if "position" in style or "long-term" in style or "leap" in style:
        dte_range = (180, 900)
        delta_range = (0.5, 0.85)
    elif "medium" in style:
        dte_range = (30, 120)
        delta_range = (0.4, 0.7)
    else:
        dte_range = (7, 60)
        delta_range = (0.25, 0.6)

    # Conviction tilt: stronger signals/ratings can take a bit more aggression
    strong = "strong" in sig or tr >= 22
    weak = ("avoid" in sig) or tr < 12

    if strong:
        dte_range = (max(5, int(dte_range[0] * 0.8)), int(dte_range[1] * 0.9))
        delta_range = (max(0.2, delta_range[0] * 0.9), max(delta_range[0] * 0.9, delta_range[1] * 0.95))
    elif weak:
        dte_range = (int(dte_range[0] * 1.2), int(dte_range[1] * 1.3))
        delta_range = (min(0.7, delta_range[0] * 1.1), min(0.9, delta_range[1] * 1.1))

    return dte_range, delta_range


def _parse_dte(details: dict) -> Optional[int]:
    dte = details.get("days_to_expiration")
    if isinstance(dte, (int, float)):
        return int(dte)
    exp = details.get("expiration_date")
    if not exp:
        return None
    try:
        exp_dt = dt.date.fromisoformat(str(exp))
        return (exp_dt - dt.date.today()).days
    except Exception:
        return None


def _moneyness(contract_type: str, strike: Optional[float], underlying: Optional[float]) -> Optional[float]:
    if strike is None or underlying is None or underlying <= 0:
        return None
    if contract_type == "call":
        return (strike - underlying) / underlying
    return (underlying - strike) / underlying


def _normalize_contract(raw: dict, underlying_price: Optional[float]) -> Optional[dict]:
    details = raw.get("details") or {}
    ticker = details.get("ticker") or raw.get("ticker")
    if not ticker:
        return None

    contract_type = (details.get("contract_type") or raw.get("contract_type") or "").lower()
    if contract_type not in {"call", "put"}:
        return None

    strike = details.get("strike_price")
    dte = _parse_dte(details)

    greeks = raw.get("greeks") or {}
    delta = greeks.get("delta")

    last_quote = raw.get("last_quote") or {}
    bid = last_quote.get("bid")
    ask = last_quote.get("ask")
    mid = last_quote.get("midpoint")
    if mid is None and bid is not None and ask is not None:
        mid = (bid + ask) / 2.0

    # Require a usable quote
    if bid is None or ask is None or mid is None:
        return None

    last_trade = raw.get("last_trade") or raw.get("last") or {}
    last_price = None
    if isinstance(last_trade, dict):
        last_price = last_trade.get("price") or last_trade.get("p")

    iv = raw.get("implied_volatility") or raw.get("iv")
    open_interest = raw.get("open_interest")
    volume = (raw.get("day") or {}).get("volume") or raw.get("volume")
    break_even = raw.get("break_even_price") or raw.get("break_even")

    underlying_obj = raw.get("underlying_asset") or {}
    underlying_last = (
        underlying_obj.get("last") or underlying_obj.get("price") or underlying_obj.get("last_price")
    )
    underlying_px = underlying_price or underlying_last

    spread_pct = None
    if bid is not None and ask is not None and bid > 0:
        mid_calc = mid if mid is not None else (bid + ask) / 2.0
        if mid_calc:
            spread_pct = (ask - bid) / mid_calc

    return {
        "ticker": ticker,
        "contract_type": contract_type,
        "strike": strike,
        "expiration": details.get("expiration_date"),
        "dte": dte,
        "delta": delta,
        "iv": iv,
        "open_interest": open_interest,
        "volume": volume,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "last": last_price,
        "spread_pct": spread_pct,
        "breakeven": break_even,
        "underlying": underlying_px,
        "moneyness": _moneyness(contract_type, strike, underlying_px),
    }


def _score_option(
    option: dict,
    dte_range: Tuple[int, int],
    delta_range: Tuple[float, float],
    trade_style: str,
    tech_rating: Optional[float],
    risk_score: Optional[float],
    price_target: Optional[float],
) -> float:
    score = 0.0

    oi = option.get("open_interest") or 0
    vol = option.get("volume") or 0
    dte = option.get("dte") or 0
    spread = option.get("spread_pct")
    delta = option.get("delta")
    iv = option.get("iv")
    mny = option.get("moneyness")
    breakeven = option.get("breakeven")
    strike = option.get("strike")

    # Liquidity
    if oi >= 500:
        score += 2.0
    if oi >= 2000:
        score += 1.0
    if vol >= 100:
        score += 1.0
    if vol >= 500:
        score += 1.0

    # Spread tightness
    if spread is not None:
        if spread <= 0.05:
            score += 3.0
        elif spread <= 0.1:
            score += 2.0
        elif spread <= 0.15:
            score += 1.0
        else:
            score -= 2.0

    # Delta alignment
    if delta is not None:
        delta_abs = abs(delta)
        low, high = delta_range
        if low <= delta_abs <= high:
            mid = (low + high) / 2.0
            span = max(high - low, 0.01)
            score += 3.0 * (1 - abs(delta_abs - mid) / span)
        else:
            score -= 0.5

    # DTE fit
    lo_dte, hi_dte = dte_range
    if dte > 0:
        if dte < lo_dte:
            score -= 1.0
        elif dte > hi_dte:
            score -= 0.5
        else:
            mid = (lo_dte + hi_dte) / 2.0
            span = max(hi_dte - lo_dte, 1)
            score += 2.0 * (1 - abs(dte - mid) / span)

    # Moneyness preference (ATM to slight OTM)
    if mny is not None:
        if mny <= 0.05:
            score += 1.5
        elif mny <= 0.12:
            score += 1.0
        elif mny > 0.25:
            score -= 0.5

    # IV preference: reward moderate or cheap IV
    if iv is not None:
        if iv <= 0.25:
            score += 1.0
        elif iv <= 0.6:
            score += 0.5
        elif iv >= 1.0:
            score -= 0.5

    # Stock quality bonus
    if tech_rating is not None and not math.isnan(tech_rating):
        score += min(3.0, max(0.0, tech_rating) / 6.0)
    if risk_score is not None and not math.isnan(risk_score):
        score += (risk_score - 0.5) * 2.0

    # Target alignment bonus: breakeven near target for directional trades
    if price_target is not None and breakeven is not None and strike is not None:
        distance = price_target - breakeven if option["contract_type"] == "call" else breakeven - price_target
        if distance >= 0:
            score += 0.5

    return score


def _reason_snippet(option: dict, trade_style: str, dte_range: Tuple[int, int], delta_range: Tuple[float, float]) -> str:
    parts: List[str] = []
    dte = option.get("dte")
    delta = option.get("delta")
    spread = option.get("spread_pct")
    oi = option.get("open_interest")
    vol = option.get("volume")
    iv = option.get("iv")

    if dte is not None:
        parts.append(f"{dte}d aligns with {trade_style} window {dte_range[0]}-{dte_range[1]}d")
    if delta is not None:
        parts.append(f"delta {delta:.2f} (target {delta_range[0]:.2f}-{delta_range[1]:.2f})")
    if oi is not None:
        parts.append(f"OI {oi:,}")
    if vol is not None:
        parts.append(f"vol {vol:,}")
    if spread is not None:
        parts.append(f"spread {spread*100:.1f}%")
    if iv is not None:
        parts.append(f"IV {iv:.2f}")

    return " | ".join(parts)


def select_option_candidates(
    chain: Iterable[dict],
    direction: str,
    trade_style: str,
    underlying_price: Optional[float],
    tech_rating: Optional[float],
    risk_score: Optional[float],
    price_target: Optional[float] = None,
    min_oi: int = 500,
    min_vol: int = 100,
    signal: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Filter + score an option chain and return ranked candidates.
    Direction should be 'call' or 'put'.
    """
    dte_range, delta_range = _strategy_bands(trade_style, signal, tech_rating)
    direction = (direction or "").lower()
    if direction not in {"call", "put"}:
        direction = "call"

    def _pick(threshold_oi: int, threshold_vol: int) -> List[Dict[str, Any]]:
        picks: List[Dict[str, Any]] = []
        for raw in chain:
            norm = _normalize_contract(raw, underlying_price)
            if not norm:
                continue
            if norm["contract_type"] != direction:
                continue
            if norm["open_interest"] is None or norm["open_interest"] < threshold_oi:
                continue
            if norm["volume"] is None or norm["volume"] < threshold_vol:
                continue
            if norm["dte"] is not None and norm["dte"] < 1:
                continue
            spread = norm.get("spread_pct")
            if spread is None:
                continue
            # Relaxed: allow up to 30% (vs 25%) and consider >15% only if OI is decent
            if spread > 0.30:
                continue
            if spread > 0.15 and norm.get("open_interest", 0) < max(200, threshold_oi):
                continue

            score = _score_option(
                norm,
                dte_range=dte_range,
                delta_range=delta_range,
                trade_style=trade_style,
                tech_rating=tech_rating,
                risk_score=risk_score,
                price_target=price_target,
            )
            norm["score"] = score
            norm["reason"] = _reason_snippet(norm, trade_style, dte_range, delta_range)
            picks.append(norm)
        picks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return picks

    picks = _pick(min_oi, min_vol)
    if not picks:
        # Relax filters for thin underlyings
        picks = _pick(max(50, min_oi // 5), max(10, min_vol // 5))

    return picks
