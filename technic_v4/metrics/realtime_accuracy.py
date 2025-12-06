"""Real-time prediction accuracy tracker."""

from __future__ import annotations


def update_realtime_accuracy(predictions, actuals):
    """Track near-real-time prediction success rate."""
    if len(predictions) != len(actuals) or len(predictions) == 0:
        return 0.0
    correct = sum([p == a for p, a in zip(predictions, actuals)])
    return correct / len(predictions)


__all__ = ["update_realtime_accuracy"]
