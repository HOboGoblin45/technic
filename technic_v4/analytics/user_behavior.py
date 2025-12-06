"""User behavior pattern tracker."""

from __future__ import annotations

import datetime


def track_user_behavior(user_id, action_type):
    """Log a simple user action with timestamp for clustering later."""
    return {
        "user": user_id,
        "action": action_type,
        "timestamp": datetime.datetime.utcnow(),
    }


__all__ = ["track_user_behavior"]
