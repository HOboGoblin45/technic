"""Reward model scoring based on outcome quality."""

from __future__ import annotations


def reward_score(prediction: float, actual: float) -> float:
    """Generate reward signals based on actual vs predicted outcomes."""
    delta = abs(prediction - actual)
    if delta < 0.5:
        return 1.0
    if delta < 1.5:
        return 0.5
    return -0.5


__all__ = ["reward_score"]
