"""Mid-term horizon signal processing stub."""

from __future__ import annotations

from typing import Dict, Any


def process_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
    """Apply mid-term adjustments (placeholder)."""
    signal = dict(signal)
    signal["horizon"] = "mid"
    return signal


__all__ = ["process_signal"]
