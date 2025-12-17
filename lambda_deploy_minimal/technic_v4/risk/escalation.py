"""Escalation trigger framework."""

from __future__ import annotations


def escalation_level(score: float, drawdown_risk: float) -> int:
    """Return alert level 1–5; trigger when robustness is low and drawdown risk high."""
    if score < 0.45 and drawdown_risk > 0.25:
        return 5
    if score < 0.6:
        return 3
    return 1


__all__ = ["escalation_level"]
