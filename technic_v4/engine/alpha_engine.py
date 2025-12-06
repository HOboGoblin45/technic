"""Adaptive alpha fusion utilities."""

from __future__ import annotations

from typing import Sequence


def adaptive_alpha_fusion(alpha_streams: Sequence[float], weights: Sequence[float], decay_factor: float = 0.95) -> float:
    """
    Dynamically fuse alpha signals by decaying low-confidence streams.
    """
    adjusted = [w * decay_factor if abs(a) < 0.1 else w for a, w in zip(alpha_streams, weights)]
    norm = sum(adjusted)
    return sum([a * w for a, w in zip(alpha_streams, adjusted)]) / norm if norm else 0.0


__all__ = ["adaptive_alpha_fusion"]
