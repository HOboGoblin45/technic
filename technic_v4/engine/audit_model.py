"""Self-audit model accuracy monitor."""

from __future__ import annotations

import numpy as np
from typing import Dict


def audit_predictions(preds: np.ndarray, actuals: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute basic accuracy stats and error count vs a threshold."""
    if preds.shape != actuals.shape:
        raise ValueError("preds and actuals must have the same shape")
    if preds.size == 0:
        return {"total": 0, "errors": 0, "accuracy": 0.0}
    errors = (preds > threshold) != actuals
    return {
        "total": float(len(preds)),
        "errors": float(errors.sum()),
        "accuracy": 1.0 - float(errors.sum()) / float(len(preds)),
    }


__all__ = ["audit_predictions"]
