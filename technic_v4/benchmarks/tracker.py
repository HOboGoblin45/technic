"""Auto-benchmark tracker for Technic vs common benchmarks."""

from __future__ import annotations

import pandas as pd


def calculate_alpha(signal_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Compute average excess return of signals over a benchmark."""
    excess_return = signal_returns - benchmark_returns
    return float(excess_return.mean())


__all__ = ["calculate_alpha"]
