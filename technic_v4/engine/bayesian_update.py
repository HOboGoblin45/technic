"""Bayesian belief update utility for signal likelihoods."""

from __future__ import annotations


def bayesian_update(prior: float, likelihood: float) -> float:
    """Compute posterior using a simple Bernoulli/Beta-style update."""
    denom = (prior * likelihood) + ((1 - prior) * (1 - likelihood))
    if denom == 0:
        return 0.0
    return (prior * likelihood) / denom


__all__ = ["bayesian_update"]
