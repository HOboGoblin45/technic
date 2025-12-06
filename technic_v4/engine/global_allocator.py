"""Global capital coordination dispatcher."""

from __future__ import annotations


def resolve_conflicts(trades):
    """When multiple strategies trigger, resolve conflicts by tier/time/spread."""
    return sorted(trades, key=lambda t: (-t["tier"], t["time"], t["spread"]))


__all__ = ["resolve_conflicts"]
