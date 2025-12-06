"""Causal inference helper for macro/event impact analysis."""

from __future__ import annotations

from typing import Any

try:
    import dowhy  # type: ignore

    HAVE_DOWHY = True
except ImportError:  # pragma: no cover
    HAVE_DOWHY = False


def analyze_causal_impact(df, treatment_col: str, outcome_col: str) -> Any:
    """Estimate causal effect of a treatment on an outcome using DoWhy.

    Args:
        df: DataFrame containing treatment, outcome, and confounders.
        treatment_col: Name of the treatment column.
        outcome_col: Name of the outcome column.

    Returns:
        Estimated causal effect value.
    """
    if not HAVE_DOWHY:
        raise ImportError("dowhy is required for causal impact analysis.")

    model = dowhy.CausalModel(
        data=df,
        treatment=treatment_col,
        outcome=outcome_col,
        common_causes=["volume", "volatility"],
    )
    identified = model.identify_effect()
    estimate = model.estimate_effect(
        identified, method_name="backdoor.propensity_score_matching"
    )
    return estimate.value


__all__ = ["analyze_causal_impact"]
