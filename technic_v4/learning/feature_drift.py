"""Feature drift detector using Jensen-Shannon divergence."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon


def detect_feature_drift(live: np.ndarray, train: np.ndarray, threshold: float = 0.3) -> bool:
    live_p = np.abs(live) + 1e-9
    train_p = np.abs(train) + 1e-9
    live_p = live_p / live_p.sum()
    train_p = train_p / train_p.sum()
    score = float(jensenshannon(live_p, train_p))
    return score > threshold


__all__ = ["detect_feature_drift"]
