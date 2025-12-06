"""News-based risk aversion overlay using zero-shot classification.

Provides a simple helper to score headlines for "risk" tone. If transformers
are unavailable, returns a neutral default score.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

try:
    from transformers import pipeline

    HAVE_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    HAVE_TRANSFORMERS = False


@lru_cache(maxsize=1)
def _get_classifier():
    if not HAVE_TRANSFORMERS:
        return None
    return pipeline("zero-shot-classification")


def score_headline(headline: str) -> float:
    """Score a headline's risk tone; higher means more risk-averse sentiment.

    Args:
        headline: News headline text.

    Returns:
        Risk score in [0,1] if model available; otherwise 0.0 as neutral.
    """
    clf = _get_classifier()
    if clf is None:
        return 0.0

    labels = ["risk", "opportunity", "neutral"]
    result = clf(headline, candidate_labels=labels)
    try:
        idx = result["labels"].index("risk")
        return float(result["scores"][idx])
    except (KeyError, ValueError, IndexError):
        return 0.0


__all__ = ["score_headline"]
