"""Execution slippage simulator."""

from __future__ import annotations


def estimate_slippage(entry_price: float, liquidity_factor: float = 0.002) -> float:
    """Estimate slippage-adjusted entry price."""
    return entry_price * (1 + liquidity_factor)


__all__ = ["estimate_slippage"]
