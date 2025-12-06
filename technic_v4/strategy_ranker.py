"""Strategy re-weighting via feedback."""

from __future__ import annotations


def apply_feedback_weight(base_score: float, feedback_score: float) -> float:
    return base_score * (1 + feedback_score * 0.2)


__all__ = ["apply_feedback_weight"]
