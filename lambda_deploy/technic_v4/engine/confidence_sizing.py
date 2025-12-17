"""Confidence-based position sizing utilities.

Weights assets by inverse ensemble prediction uncertainty: higher variance
across ensemble members leads to smaller weight.
"""

from __future__ import annotations

import numpy as np


def size_by_confidence(ensemble_preds: np.ndarray) -> np.ndarray:
    """Compute normalized weights from ensemble predictions.

    Args:
        ensemble_preds: Array shaped (n_models, n_assets) with per-model
            predictions for each asset.

    Returns:
        A 1-D numpy array of normalized weights summing to 1.
    """
    if ensemble_preds.ndim != 2:
        raise ValueError("ensemble_preds must be 2-D (n_models, n_assets)")
    means = np.mean(ensemble_preds, axis=0)
    stds = np.std(ensemble_preds, axis=0)
    # Avoid zero division for perfectly aligned ensemble predictions.
    inv_risk = 1.0 / (stds + 1e-6)
    weights = inv_risk / inv_risk.sum() if inv_risk.sum() != 0 else np.full_like(inv_risk, 1.0 / inv_risk.size)
    return weights


__all__ = ["size_by_confidence"]
