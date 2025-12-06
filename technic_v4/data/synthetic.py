"""Synthetic turbulence generator for stress testing."""

from __future__ import annotations

import numpy as np


def create_turbulence(prices: np.ndarray, volumes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Introduce volume anomalies, spread gaps, and correlated vol bursts."""
    noisy_prices = prices.copy()
    noisy_volumes = volumes.copy()

    # Volume anomaly: random spikes
    spikes = np.random.choice(len(volumes), size=max(1, len(volumes) // 10), replace=False)
    noisy_volumes[spikes] *= 2.0

    # Spread gaps: occasional jumps/drops
    gaps = np.random.choice(len(prices), size=max(1, len(prices) // 15), replace=False)
    noisy_prices[gaps] *= np.random.choice([0.97, 1.03], size=len(gaps))

    # Correlated volatility bursts
    burst = np.random.normal(1.0, 0.05, size=len(prices))
    noisy_prices *= burst
    return noisy_prices, noisy_volumes


__all__ = ["create_turbulence"]
