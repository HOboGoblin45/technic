"""Portfolio-level impact calculator."""

from __future__ import annotations


def portfolio_impact(signal, portfolio):
    """Estimate expected impact of signal execution on overall metrics."""
    weight = signal["weight"]
    risk = signal["volatility"]
    exposure = sum([a["weight"] for a in portfolio])
    return weight * (1 - risk) * (1 - exposure)


__all__ = ["portfolio_impact"]
