"""Execution delay tolerance filter."""

from __future__ import annotations


def assess_latency_impact(entry_price: float, delay_seconds: float, volatility: float) -> float:
    """Estimate slippage impact from execution delay and volatility."""
    slippage = entry_price * (1 + (volatility * delay_seconds / 100))
    return slippage


__all__ = ["assess_latency_impact"]
