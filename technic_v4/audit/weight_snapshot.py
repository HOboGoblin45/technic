"""Model weights snapshot storage for backtracking."""

from __future__ import annotations


def snapshot_model_weights(model_id, weights):
    """Store model weights and thresholds alongside signal IDs."""
    return {"model_id": model_id, "weights": dict(weights)}


__all__ = ["snapshot_model_weights"]
