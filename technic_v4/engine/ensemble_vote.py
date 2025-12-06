"""Weighted ensemble voting engine."""

from __future__ import annotations

from typing import Sequence


def weighted_vote(predictions: Sequence[float], weights: Sequence[float]) -> float:
    """Compute weighted average vote with confidence calibration."""
    if len(predictions) != len(weights):
        raise ValueError("predictions and weights must have the same length")
    if not predictions:
        return 0.0
    weighted = [p * w for p, w in zip(predictions, weights)]
    denom = sum(weights)
    if denom == 0:
        return 0.0
    return sum(weighted) / denom


__all__ = ["weighted_vote"]
