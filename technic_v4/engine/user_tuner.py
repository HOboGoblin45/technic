"""Personalized strategy tuner to modulate weights by user risk tolerance."""

from __future__ import annotations

from typing import Sequence, List


def adjust_weights(base_weights: Sequence[float], risk_factor: float) -> List[float]:
    """Scale base weights by a user risk factor."""
    return [w * risk_factor for w in base_weights]


__all__ = ["adjust_weights"]
