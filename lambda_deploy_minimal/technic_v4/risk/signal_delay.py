"""Auto-gating signal delay/block logic."""

from __future__ import annotations


def delay_or_block_signal(signal):
    """Delay or discard signals that fail the internal quality bar."""
    if signal["confidence"] < 0.6:
        return "DELAY"
    if signal["spread"] > 0.05:
        return "BLOCK"
    return "SEND"


__all__ = ["delay_or_block_signal"]
