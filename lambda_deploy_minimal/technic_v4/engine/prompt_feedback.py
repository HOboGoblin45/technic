"""Smart prompt refinement feedback for query precision."""

from __future__ import annotations


def suggest_improvements(user_query: str, history: str) -> str:
    """Provide simple heuristic suggestions to refine user queries."""
    if "breakout" in user_query.lower() and "trend" in history.lower():
        return "Consider specifying timeframe for breakout (e.g., 5-day high)."
    return "Query OK"


__all__ = ["suggest_improvements"]
