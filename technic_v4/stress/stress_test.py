"""Adversarial scenario stress testing for macro events."""

from __future__ import annotations

from typing import Iterable
import numpy as np


def generate_adversarial_scenarios(data_stream: np.ndarray, events: Iterable[str]) -> np.ndarray:
    """Mutate market data streams with edge-case macro events."""
    stressed = data_stream.copy()
    for event in events:
        if event == "flash_crash":
            stressed *= 0.8
        elif event == "taper_shock":
            stressed *= 0.9
        elif event == "volatility_burst":
            stressed *= 1.1
    return stressed


__all__ = ["generate_adversarial_scenarios"]
