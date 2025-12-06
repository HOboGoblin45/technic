"""Bayesian model selector using BIC/posterior likelihood."""

from __future__ import annotations

import math
from typing import List, Dict, Tuple


def bic(log_likelihood: float, num_params: int, num_samples: int) -> float:
    """Bayesian Information Criterion (lower is better)."""
    return -2 * log_likelihood + num_params * math.log(max(num_samples, 1))


def choose_model(models: List[Dict]) -> Tuple[str, float]:
    """
    Dynamically choose best-performing model for current conditions.

    Each model dict should include:
      - name
      - log_likelihood
      - num_params
      - num_samples
    """
    scores = []
    for m in models:
        b = bic(m["log_likelihood"], m["num_params"], m["num_samples"])
        scores.append((m["name"], b))
    best = min(scores, key=lambda x: x[1])
    # Posterior weights (softmax over negative BIC)
    denom = sum(math.exp(-0.5 * (b - best[1])) for _, b in scores)
    posterior = math.exp(-0.5 * (best[1] - best[1])) / denom if denom else 0.0
    return best[0], posterior


__all__ = ["bic", "choose_model"]
