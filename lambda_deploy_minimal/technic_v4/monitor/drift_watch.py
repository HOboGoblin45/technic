"""Real-time model drift watcher using population stability index (PSI)."""

from __future__ import annotations

import numpy as np


def population_stability_index(expected, actual, bins: int = 10) -> float:
    """Compute PSI between expected and actual feature distributions."""
    expected_hist, _ = np.histogram(expected, bins=bins)
    actual_hist, _ = np.histogram(actual, bins=bins)
    psi = np.sum(
        (expected_hist - actual_hist)
        * np.log((expected_hist + 1e-6) / (actual_hist + 1e-6))
    )
    return float(psi)


__all__ = ["population_stability_index"]
