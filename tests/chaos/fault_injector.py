"""Fault injection chaos tests (simulation stubs)."""

from __future__ import annotations

import time


def simulate_latency(seconds: float = 1.0):
    time.sleep(min(seconds, 2.0))


def simulate_model_failure():
    raise RuntimeError("Simulated model failure")


def simulate_memory_exhaustion():
    try:
        _ = [0] * 10_000  # small placeholder
    except MemoryError:
        return True
    return False


__all__ = ["simulate_latency", "simulate_model_failure", "simulate_memory_exhaustion"]
