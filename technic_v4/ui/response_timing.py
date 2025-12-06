"""Auto-tuned UI element timing based on latency."""

from __future__ import annotations


def dynamic_ui_delay(ping_ms: float) -> float:
    """Adjust loading indicators/tooltips/chart updates based on latency."""
    return min(max(ping_ms / 1000.0, 0.2), 2.5)


__all__ = ["dynamic_ui_delay"]
