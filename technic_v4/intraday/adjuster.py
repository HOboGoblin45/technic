"""Intraday adjustment engine using high-frequency data."""

from __future__ import annotations

from typing import Dict, List


def adjust_weights(positions: List[Dict], hf_vol: float, catalyst: bool = False) -> List[Dict]:
    """Re-weight positions mid-day for volatility or catalyst updates."""
    adj = []
    for pos in positions:
        w = pos.get("weight", 0.0)
        # Reduce weight if high intraday vol or catalyst present
        if hf_vol > 2.0 or catalyst:
            w *= 0.9
        adj_pos = dict(pos)
        adj_pos["weight"] = w
        adj.append(adj_pos)
    return adj


__all__ = ["adjust_weights"]
