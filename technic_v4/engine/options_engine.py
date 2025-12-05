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
