"""Signal lifecycle tracker."""

from __future__ import annotations


def track_lifecycle(signal_id, stage, timestamp):
    """Attach lifecycle metadata to each signal."""
    return {"id": signal_id, "stage": stage, "time": timestamp}


__all__ = ["track_lifecycle"]
