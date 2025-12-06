"""Robustness metrics engine."""

from __future__ import annotations

from typing import Dict


def robustness_score(strategy: Dict) -> float:
    """Compute robustness score using Sharpe decay, drawdown sensitivity, tail-risk."""
    sharpe_decay = strategy.get("sharpe_decay", 0.0)
    drawdown_sensitivity = strategy.get("drawdown_sensitivity", 0.0)
    tail_risk = strategy.get("tail_risk", 0.0)
    # Lower drawdown/tail risk is better, so invert those components.
    score = max(0.0, 1.0 - (drawdown_sensitivity + tail_risk)) + max(0.0, 1.0 - sharpe_decay)
    return score / 2.0


__all__ = ["robustness_score"]
