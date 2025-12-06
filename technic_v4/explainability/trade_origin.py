"""Trade origin explanation helper."""

from __future__ import annotations


def explain_trade(origin_data):
    """Map a trade back to its originating model, features, and reason."""
    return {
        "model": origin_data.get("model"),
        "top_features": origin_data.get("features", [])[:3],
        "reason": origin_data.get("signal_reason"),
    }


__all__ = ["explain_trade"]
