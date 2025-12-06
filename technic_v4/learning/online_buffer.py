"""Adaptive replay buffer with recency/vol weighting and decay pruning."""

from __future__ import annotations

from typing import List, Tuple


class OnlineBuffer:
    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.samples: List[Tuple[dict, float]] = []

    def add(self, sample: dict, volatility: float, weight: float = 1.0):
        """Weight samples by recency and volatility."""
        self.samples.append((sample, weight / max(volatility, 1e-3)))
        self._prune()

    def _prune(self):
        # Apply decay and drop low-weight samples
        pruned = []
        for s, w in self.samples:
            w *= self.decay
            if w > 1e-3:
                pruned.append((s, w))
        self.samples = pruned


__all__ = ["OnlineBuffer"]
