"""UI signal overlay panel renderer."""

from __future__ import annotations


def render_signal_overlay(signal_data) -> str:
    """Display signal confidence, sector exposure, and return potential inline."""
    return (
        f"Confidence: {signal_data['confidence']:.2f}, "
        f"Sector: {signal_data['sector']}, "
        f"Expected Return: {signal_data['return_estimate']}%"
    )


__all__ = ["render_signal_overlay"]
