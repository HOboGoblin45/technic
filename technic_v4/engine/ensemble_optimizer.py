"""Dynamic reweighting of model ensemble based on recent performance."""

from __future__ import annotations

from typing import Dict, Any


def reweight_ensemble(models, performance_window: Dict[Any, float]):
    """Boost or diminish model voting power based on short-term performance."""
    total = sum(performance_window.values())
    if total == 0:
        return {m: 0 for m in models}
    return {m: performance_window.get(m, 0.0) / total for m in models}


__all__ = ["reweight_ensemble"]
