"""Signal curator with priority queue selection."""

from __future__ import annotations

import heapq
from typing import Sequence


def curate_signals(signal_batch: Sequence[dict], top_n: int = 5):
    """Rank signals and select top N for execution routing."""
    return heapq.nlargest(top_n, signal_batch, key=lambda x: x["score"])


__all__ = ["curate_signals"]
