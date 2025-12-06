"""Adversarial strategy simulation for shock events and fake news."""

from __future__ import annotations

import numpy as np


def simulate_shock(event_type: str, baseline_data: np.ndarray) -> np.ndarray:
    """Simulate how alpha models react to shock events/low liquidity/fake news."""
    data = baseline_data.copy()
    if event_type == "flash_crash":
        data *= 0.9
    elif event_type == "short_squeeze":
        data *= 1.2
    elif event_type == "fake_news":
        data *= 0.95
    return data


__all__ = ["simulate_shock"]
