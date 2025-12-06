"""Emotional sentiment guardrails to flag impulsive behavior."""

from __future__ import annotations


def detect_sentiment_drift(recent_actions):
    """Detect overly emotional trading behavior and issue soft warnings."""
    if recent_actions.count("panic_sell") > 2:
        return "Warning: Emotional Trading Detected"
    return None


__all__ = ["detect_sentiment_drift"]
