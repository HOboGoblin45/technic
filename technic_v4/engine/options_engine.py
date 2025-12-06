"""
Option scoring layer: pick best contracts for a bullish/bearish view on the underlying.
Designed to work with Polygon Massive data (IV/Greeks) but degrades gracefully if absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import numpy as np


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
    expected_value: float | None = None
    expected_return_pct: float | None = None
    risk_score: float | None = None
    term_slope: float | None = None
    skew: float | None = None
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
    underlying_price = float(chain.get("underlying_price", pd.Series([np.nan])).iloc[0]) if "underlying_price" in chain else np.nan
    mean_iv = float(chain["iv"].dropna().mean()) if "iv" in chain else np.nan
    days_to_exp = float(chain["days_to_exp"].median()) if "days_to_exp" in chain else 30.0
    # Term structure proxy: compare near vs far expiries
    term_slope = None
    try:
        near = chain.sort_values("days_to_exp").groupby("expiration").head(1)
        far = chain.sort_values("days_to_exp", ascending=False).groupby("expiration").head(1)
        near_iv = near["iv"].mean()
        far_iv = far["iv"].mean()
        term_slope = float(far_iv - near_iv) if pd.notna(near_iv) and pd.notna(far_iv) else None
    except Exception:
        term_slope = None

    if dir_up:
        # Long call near 0.3-0.5 delta
        calls = chain[chain["option_type"].str.lower() == "call"]
        if not calls.empty:
            calls["delta_abs"] = calls["delta"].abs()
            long_call = calls.iloc[(calls["delta_abs"] - 0.4).abs().argsort()[:1]]
            for _, row in long_call.iterrows():
                rr_est = target_rr
                prob = max(0.0, 1 - abs(row.get("delta", 0)))
                mid = float((row.get("bid_price", 0) + row.get("ask_price", 0)) / 2)
                leg = {"side": "buy", "type": "call", "strike": row["strike"], "expiry": str(row["expiration"]), "premium": mid}
                strat = OptionStrategy(
                    symbol=symbol,
                    strategy_type="Long Call",
                    legs=[leg],
                    underlying_alpha=alpha_score,
                    iv_rank=iv_rank,
                    rr_estimate=rr_est,
                    prob_profit_estimate=prob,
                    term_slope=term_slope,
                    notes="OTM call targeting upside move",
                )
                strat.expected_value, strat.expected_return_pct, strat.risk_score = estimate_strategy_ev(
                    strat, underlying_price, mean_iv, alpha_score, days_to_exp=days_to_exp
                )
                # Notes about IV regime
                if iv_rank is not None:
                    if iv_rank >= 0.8:
                        strat.notes += "; High IV (consider spreads / selling)"
                    elif iv_rank <= 0.2:
                        strat.notes += "; Low IV (favor buying)"
                out.append(strat)
        # Bull call spread: buy ~0.35 delta, sell higher strike
        if not calls.empty:
            calls_sorted = calls.sort_values("strike")
            lower = calls_sorted.iloc[(calls_sorted["delta_abs"] - 0.35).abs().argsort()[:1]]
            upper = calls_sorted.iloc[(calls_sorted["delta_abs"] - 0.15).abs().argsort()[:1]]
            if not lower.empty and not upper.empty:
                l = lower.iloc[0]
                u = upper.iloc[0]
                mid_lower = float((l.get("bid_price", 0) + l.get("ask_price", 0)) / 2)
                mid_upper = float((u.get("bid_price", 0) + u.get("ask_price", 0)) / 2)
                legs = [
                    {"side": "buy", "type": "call", "strike": l["strike"], "expiry": str(l["expiration"]), "premium": mid_lower},
                    {"side": "sell", "type": "call", "strike": u["strike"], "expiry": str(u["expiration"]), "premium": mid_upper},
                ]
                rr_est = target_rr * 0.8
                prob = max(0.0, 1 - abs(l.get("delta", 0)))
                strat = OptionStrategy(
                    symbol=symbol,
                    strategy_type="Bull Call Spread",
                    legs=legs,
                    underlying_alpha=alpha_score,
                    iv_rank=iv_rank,
                    rr_estimate=rr_est,
                    prob_profit_estimate=prob,
                    term_slope=term_slope,
                    notes="Defined risk vertical for upside exposure",
                )
                strat.expected_value, strat.expected_return_pct, strat.risk_score = estimate_strategy_ev(
                    strat, underlying_price, mean_iv, alpha_score, days_to_exp=days_to_exp
                )
                if iv_rank is not None:
                    if iv_rank >= 0.8:
                        strat.notes += "; High IV (credit spreads favorable)"
                    elif iv_rank <= 0.2:
                        strat.notes += "; Low IV (debit spreads acceptable)"
                out.append(strat)
        # Cash-secured put skeleton (scored, not executed)
        puts = chain[chain["option_type"].str.lower() == "put"]
        if not puts.empty:
            puts["delta_abs"] = puts["delta"].abs()
            csp = puts.iloc[(puts["delta_abs"] - 0.2).abs().argsort()[:1]]
            for _, row in csp.iterrows():
                prob = max(0.0, 1 - abs(row.get("delta", 0)))
                mid = float((row.get("bid_price", 0) + row.get("ask_price", 0)) / 2)
                legs = [{"side": "sell", "type": "put", "strike": row["strike"], "expiry": str(row["expiration"]), "premium": mid}]
                strat = OptionStrategy(
                    symbol=symbol,
                    strategy_type="Cash-Secured Put",
                    legs=legs,
                    underlying_alpha=alpha_score,
                    iv_rank=iv_rank,
                    rr_estimate=target_rr * 0.6,
                    prob_profit_estimate=prob,
                    term_slope=term_slope,
                    notes="Income approach aligned with bullish bias",
                )
                strat.expected_value, strat.expected_return_pct, strat.risk_score = estimate_strategy_ev(
                    strat, underlying_price, mean_iv, alpha_score, days_to_exp=days_to_exp
                )
                if iv_rank is not None and iv_rank >= 0.8:
                    strat.notes += "; High IV (selling premium attractive)"
                out.append(strat)

    return out


def estimate_strategy_ev(
    strategy: OptionStrategy,
    current_price: float,
    iv: float,
    alpha_score: float,
    days_to_exp: float = 30.0,
    n_sims: int = 5000,
    risk_free: float = 0.01,
) -> tuple[float, float, float]:
    """
    Rough expected value / return / risk_score estimate using simplified payoff math.
    - expected move derived from alpha_score (heuristic: 1% per alpha point, clipped)
    - iv used to penalize high-vol scenarios in risk_score
    - payoff approximated on a single expected price at expiry
    """
    if current_price is None or np.isnan(current_price) or days_to_exp <= 0:
        return (np.nan, np.nan, np.nan)
    vol = abs(iv) if not np.isnan(iv) else 0.4
    T = max(days_to_exp, 1) / 252.0
    # Monte Carlo for payoff distribution
    z = np.random.normal(size=n_sims)
    drift = (risk_free - 0.5 * vol * vol) * T
    diffusion = vol * np.sqrt(T) * z
    ST = current_price * np.exp(drift + diffusion)
    total_cost = 0.0
    payoffs = np.zeros(n_sims)
    for leg in strategy.legs:
        side = leg.get("side", "buy").lower()
        opt_type = leg.get("type", "call").lower()
        strike = float(leg.get("strike", 0) or 0)
        premium = float(leg.get("premium", 0) or 0)
        qty = -1 if side == "sell" else 1
        if opt_type == "call":
            intrinsic = np.maximum(ST - strike, 0)
        else:
            intrinsic = np.maximum(strike - ST, 0)
        payoffs += qty * intrinsic * 100
        total_cost += qty * premium * 100
    ev = float(payoffs.mean() - total_cost)
    pop = float((payoffs - total_cost > 0).mean())
    denom = abs(total_cost) if abs(total_cost) > 1e-6 else current_price * 100
    expected_return_pct = ev / denom if denom != 0 else np.nan
    risk_score = vol * (1 + len(strategy.legs) * 0.1)
    return (ev, expected_return_pct, risk_score if not np.isnan(risk_score) else None)


def rank_option_strategies(strategies: List[OptionStrategy]) -> List[OptionStrategy]:
    """
    Rank strategies by expected value, then expected return %, then lower risk_score.
    """
    return sorted(
        strategies,
        key=lambda s: (
            float("-inf") if s.expected_value is None or np.isnan(s.expected_value) else -s.expected_value,
            float("-inf")
            if s.expected_return_pct is None or np.isnan(s.expected_return_pct)
            else -s.expected_return_pct,
            float("inf") if s.risk_score is None or np.isnan(s.risk_score) else s.risk_score,
        ),
    )


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
        chain_df = pd.DataFrame(chain) if not isinstance(chain, pd.DataFrame) else chain
        strat = generate_option_strategies(
            sym,
            chain_df,
            alpha_score,
            tech_rating,
            target_rr,
            risk_settings,
        )
        ranked = rank_option_strategies(strat)
        # keep top 3 per symbol to limit output size
        strategies.extend(ranked[:3])
    return strategies
