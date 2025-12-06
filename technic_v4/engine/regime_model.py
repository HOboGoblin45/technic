"""Regime detection via simple market clustering heuristics."""

from __future__ import annotations


def detect_regime(volatility: float, trend_strength: float) -> str:
    """Classify macro environment into momentum/volatile/sideways/stable."""
    if volatility > 0.8 and trend_strength > 0.7:
        return "Momentum"
    if volatility > 0.8:
        return "Volatile"
    if trend_strength < 0.3:
        return "Sideways"
    return "Stable"


__all__ = ["detect_regime"]
