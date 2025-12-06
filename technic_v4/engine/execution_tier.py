"""Capital efficiency tiering for trades."""

from __future__ import annotations


def tier_trade(score: float) -> str:
    """Tag each trade with a capital-efficiency score."""
    if score > 0.9:
        return "Tier 1: High Conviction"
    if score > 0.7:
        return "Tier 2: Strong"
    return "Tier 3: Normal"


__all__ = ["tier_trade"]
