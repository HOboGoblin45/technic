"""Model explainability overlay for signal dashboard."""

from __future__ import annotations


def render_xai_overlay(signal):
    """Visualize top feature attributions on the signal dashboard."""
    features = signal["attributions"]
    top = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
    if len(top) >= 2:
        return f"Driven by {top[0][0]} and {top[1][0]}"
    if top:
        return f"Driven by {top[0][0]}"
    return "No attributions available"


__all__ = ["render_xai_overlay"]
