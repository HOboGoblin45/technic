"""Role-based model access control."""

from __future__ import annotations

ACCESS_LEVELS = ("viewer", "analyst", "developer", "admin")


def check_access(role: str, action: str) -> bool:
    """Verify access levels before exports/invocations/retraining."""
    if role not in ACCESS_LEVELS:
        return False
    if action in {"export", "retrain"}:
        return role in {"developer", "admin"}
    if action in {"invoke"}:
        return role in {"analyst", "developer", "admin"}
    return True


__all__ = ["check_access", "ACCESS_LEVELS"]
