"""Institutional API mode toggle and output helpers."""

from __future__ import annotations


def get_institutional_output(signal):
    """Enable external fund-level access to signals/predictions."""
    return {
        "timestamp": signal["time"],
        "id": signal["id"],
        "score": signal["confidence"],
        "rationale": signal["attributions"],
    }


__all__ = ["get_institutional_output"]
