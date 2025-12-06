"""Realtime volatility heat sensor."""

from __future__ import annotations

from numpy import std


def generate_volatility_heat_signal(data, timeframe):
    vol = std(data[-timeframe:])
    if vol > 2.5:
        return "🔥"
    elif vol > 1.5:
        return "🌡️"
    else:
        return "🧊"


__all__ = ["generate_volatility_heat_signal"]
