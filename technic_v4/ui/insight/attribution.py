"""Visual prediction attribution layer for UI."""

from __future__ import annotations


def show_attribution(features) -> str:
    """Reveal top contributing features for a prediction."""
    top = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    return "Top features: " + ", ".join([f"{k} ({v:.2f})" for k, v in top])


__all__ = ["show_attribution"]
