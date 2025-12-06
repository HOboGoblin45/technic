"""Model router for escalation pathways."""

from __future__ import annotations


def route_signal(signal):
    """Route edge-case signals to deeper models when shallow logic fails."""
    if signal["complexity"] > 0.8 or signal["disagreement"] > 0.5:
        return "deep_model_stack"
    return "default_model"


__all__ = ["route_signal"]
