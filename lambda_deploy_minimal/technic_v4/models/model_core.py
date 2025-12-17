"""Core model utilities including self-tuning priors."""

from __future__ import annotations

from typing import Dict


def tune_priors(prior_weights: Dict[str, float], ranking_errors: Dict[str, float]) -> Dict[str, float]:
    """Calibrate prior weights based on sector/timeframe success rates."""
    updated = {}
    for key, weight in prior_weights.items():
        error = ranking_errors.get(key, 0.0)
        # Reduce weight if errors are high; modestly increase if low.
        adjustment = 1.0 - min(max(error, -0.5), 0.5)
        updated[key] = max(0.0, weight * adjustment)
    # Renormalize
    total = sum(updated.values()) or 1.0
    for k in updated:
        updated[k] /= total
    return updated


__all__ = ["tune_priors"]
