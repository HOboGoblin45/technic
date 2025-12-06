"""Multi-device session sync detection."""

from __future__ import annotations


def detect_multi_session(devices) -> bool:
    """Detect if user has multiple sessions open and trigger sync protocol."""
    return len(set(devices)) > 1


__all__ = ["detect_multi_session"]
