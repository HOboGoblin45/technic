"""Closed-loop feedback replay buffer for reinforcement modules."""

from __future__ import annotations


class FeedbackReplayBuffer:
    """Retain historical predictions and outcomes for training RL modules."""

    def __init__(self, capacity: int = 10000):
        self.buffer = []
        self.capacity = capacity

    def add(self, signal, result) -> None:
        """Add a (signal, result) pair; evict oldest when capacity exceeded."""
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((signal, result))


__all__ = ["FeedbackReplayBuffer"]
