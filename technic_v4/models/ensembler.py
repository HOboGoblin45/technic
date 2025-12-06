"""Cross-validation ensembling helpers."""

from __future__ import annotations

import numpy as np
from typing import Sequence


def blend_predictions(pred_list: Sequence[np.ndarray]) -> np.ndarray:
    """Blend predictions from multiple folds/models by simple average."""
    if not pred_list:
        return np.array([])
    return np.mean(pred_list, axis=0)


__all__ = ["blend_predictions"]
