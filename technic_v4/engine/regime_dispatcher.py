"""Regime-specific model activation dispatcher."""

from __future__ import annotations


def choose_model_for_regime(regime: str) -> str:
    """Switch prediction models dynamically based on detected market regime."""
    if regime == "Momentum":
        return "momentum_ensemble"
    if regime == "Volatile":
        return "mean_reversion_model"
    return "general_forecaster"


__all__ = ["choose_model_for_regime"]
