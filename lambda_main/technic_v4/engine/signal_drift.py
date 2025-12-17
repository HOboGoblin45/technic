"""Signal drift monitoring via cosine distance."""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm


def drift_score(recent_signal_vec: np.ndarray, training_baseline: np.ndarray) -> float:
    """Compute cosine distance between recent and baseline feature signals."""
    if recent_signal_vec.shape != training_baseline.shape:
        raise ValueError("Vectors must have the same shape for drift scoring.")
    a = recent_signal_vec.astype(float)
    b = training_baseline.astype(float)
    denom = (norm(a) * norm(b)) or 1.0
    cosine_sim = float(np.dot(a, b) / denom)
    return 1.0 - cosine_sim


__all__ = ["drift_score"]
