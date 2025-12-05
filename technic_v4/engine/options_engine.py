"""
Option scoring layer: pick best contracts for a bullish/bearish view on the underlying.
Designed to work with Polygon Massive data (IV/Greeks) but degrades gracefully if absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class OptionPick:
    symbol: str
    contract: str
    expiry: str
    strike: float
    contract_type: str  # call/put
    delta: float
    iv: float
    bid: float
    ask: float
    score: float


@dataclass
class OptionStrategy:
    symbol: str
    strategy_type: str
    legs: list  # list of dicts {side, type, strike, expiry}
    underlying_alpha: float | None
    iv_rank: float | None
    rr_estimate: float | None
    prob_profit_estimate: float | None
    notes: str = ""


def score_options(
    chain: pd.DataFrame,
    direction: str,
    target_move_pct: float,
    max_results: int = 3,
) -> List[OptionPick]:
    """
    Score an options chain for a directional view.
    Uses a simple heuristic: favor liquid, mid-delta, reasonable IV.
    """
    if chain is None or chain.empty:
        return []

    dir_up = direction.lower().startswith("long")
    ctype = "call" if dir_up else "put"
    subset = chain[chain["option_type"].str.lower() == ctype]

    if subset.empty:
        return []

    # Heuristic scoring
    subset = subset.copy()
    subset["delta_abs"] = subset["delta"].abs()
    subset["liquidity"] = subset["open_interest"].fillna(0) + subset["volume"].fillna(0)
    subset["mid_iv_rank"] = subset["iv"].rank(pct=True)

    subset["score"] = (
        (1 - (subset["delta_abs"] - 0.5).abs()) * 0.4
        + subset["liquidity"].rank(pct=True) * 0.3
        + (1 - subset["mid_iv_rank"]) * 0.2
        + (subset["gamma"].abs().rank(pct=True)) * 0.1
    )

    picks: List[OptionPick] = []
    for _, row in subset.sort_values("score", ascending=False).head(max_results).iterrows():
        picks.append(
            OptionPick(
                symbol=row.get("underlying_symbol", ""),
                contract=row.get("symbol", ""),
                expiry=str(row.get("expiration", "")),
                strike=float(row.get("strike", 0) or 0),
                contract_type=ctype,
                delta=float(row.get("delta", 0) or 0),
                iv=float(row.get("iv", 0) or 0),
                bid=float(row.get("bid_price", 0) or 0),
                ask=float(row.get("ask_price", 0) or 0),
                score=float(row.get("score", 0) or 0),
            )
        )
    return picks


def generate_option_strategies(
    symbol: str,
    chain_snapshot: pd.DataFrame,
    alpha_score: float,
    tech_rating: float,
    target_rr: float,
    risk_settings,
) -> List[OptionStrategy]:
    """
    Generate simple bullish strategies (long call, bull call spread, cash-secured put skeleton).
    """
    if chain_snapshot is None or chain_snapshot.empty:
        return []

    # Filter expiries to ~30-90 days
    try:
        chain_snapshot["days_to_exp"] = pd.to_datetime(chain_snapshot["expiration"]) - pd.Timestamp.today()
        chain_snapshot["days_to_exp"] = chain_snapshot["days_to_exp"].dt.days
        chain = chain_snapshot[(chain_snapshot["days_to_exp"] >= 20) & (chain_snapshot["days_to_exp"] <= 120)].copy()
    except Exception:
        chain = chain_snapshot.copy()

    if chain.empty:
        return []

    out: List[OptionStrategy] = []
    iv_rank = float(chain.get("iv_rank", pd.Series([np.nan])).iloc[0]) if "iv_rank" in chain else None
    dir_up = tech_rating >= 0 and alpha_score >= 0

    if dir_up:
        # Long call near 0.3-0.5 delta
        calls = chain[chain["option_type"].str.lower() == "call"]
        if not calls.empty:
            calls["delta_abs"] = calls["delta"].abs()
            long_call = calls.iloc[(calls["delta_abs"] - 0.4).abs().argsort()[:1]]
            for _, row in long_call.iterrows():
                rr_est = target_rr
                prob = max(0.0, 1 - abs(row.get("delta", 0)))
                leg = {"side": "buy", "type": "call", "strike": row["strike"], "expiry": str(row["expiration"])}
                out.append(
                    OptionStrategy(
                        symbol=symbol,
                        strategy_type="Long Call",
                        legs=[leg],
                        underlying_alpha=alpha_score,
                        iv_rank=iv_rank,
                        rr_estimate=rr_est,
                        prob_profit_estimate=prob,
                        notes="OTM call targeting upside move",
                    )
                )
        # Bull call spread: buy ~0.35 delta, sell higher strike
        if not calls.empty:
            calls_sorted = calls.sort_values("strike")
            lower = calls_sorted.iloc[(calls_sorted["delta_abs"] - 0.35).abs().argsort()[:1]]
            upper = calls_sorted.iloc[(calls_sorted["delta_abs"] - 0.15).abs().argsort()[:1]]
            if not lower.empty and not upper.empty:
                l = lower.iloc[0]
                u = upper.iloc[0]
                legs = [
                    {"side": "buy", "type": "call", "strike": l["strike"], "expiry": str(l["expiration"])},
                    {"side": "sell", "type": "call", "strike": u["strike"], "expiry": str(u["expiration"])},
                ]
                rr_est = target_rr * 0.8
                prob = max(0.0, 1 - abs(l.get("delta", 0)))
                out.append(
                    OptionStrategy(
                        symbol=symbol,
                        strategy_type="Bull Call Spread",
                        legs=legs,
                        underlying_alpha=alpha_score,
                        iv_rank=iv_rank,
                        rr_estimate=rr_est,
                        prob_profit_estimate=prob,
                        notes="Defined risk vertical for upside exposure",
                    )
                )
        # Cash-secured put skeleton (scored, not executed)
        puts = chain[chain["option_type"].str.lower() == "put"]
        if not puts.empty:
            puts["delta_abs"] = puts["delta"].abs()
            csp = puts.iloc[(puts["delta_abs"] - 0.2).abs().argsort()[:1]]
            for _, row in csp.iterrows():
                prob = max(0.0, 1 - abs(row.get("delta", 0)))
                legs = [{"side": "sell", "type": "put", "strike": row["strike"], "expiry": str(row["expiration"])}]
                out.append(
                    OptionStrategy(
                        symbol=symbol,
                        strategy_type="Cash-Secured Put",
                        legs=legs,
                        underlying_alpha=alpha_score,
                        iv_rank=iv_rank,
                        rr_estimate=target_rr * 0.6,
                        prob_profit_estimate=prob,
                        notes="Income approach aligned with bullish bias",
                    )
                )

    return out


def score_option_strategies_for_scan(df_results: pd.DataFrame, option_chain_service, risk_settings, top_n: int = 5) -> List[OptionStrategy]:
    """
    For top N scan results, fetch option chains and generate strategies.
    """
    if df_results is None or df_results.empty:
        return []
    strategies: List[OptionStrategy] = []
    top = df_results.head(top_n)
    for _, row in top.iterrows():
        sym = row.get("Symbol")
        if not sym:
            continue
        try:
            chain, _meta = option_chain_service.fetch_chain_snapshot(symbol=sym)
        except Exception:
            continue
        alpha_score = float(row.get("AlphaScore", 0) or 0)
        tech_rating = float(row.get("TechRating", 0) or 0)
        target_rr = getattr(risk_settings, "target_rr", 2.0) if risk_settings is not None else 2.0
        strat = generate_option_strategies(
            sym,
            pd.DataFrame(chain) if not isinstance(chain, pd.DataFrame) else chain,
            alpha_score,
            tech_rating,
            target_rr,
            risk_settings,
        )
        strategies.extend(strat)
    return strategies
