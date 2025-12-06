"""Signal degradation tracking diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def compute_signal_degradation(signal_vector, benchmark_vector):
    """
    Measure correlation drop-off and volatility mismatch to detect drift.
    """
    corr = pearsonr(signal_vector[-100:], benchmark_vector[-100:])[0]
    vol_ratio = np.std(signal_vector) / np.std(benchmark_vector)
    return 1 - (corr * min(vol_ratio, 1.5))


__all__ = ["compute_signal_degradation"]
