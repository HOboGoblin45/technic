"""Interactive Brokers gateway prototype (paper trading stub)."""

from __future__ import annotations

try:
    import ib_insync  # type: ignore

    HAVE_IB = True
except ImportError:  # pragma: no cover
    HAVE_IB = False


def connect_paper():
    if not HAVE_IB:
        raise ImportError("ib_insync not installed.")
    return "ib_connection_stub"


def place_order(*args, **kwargs):
    """Stubbed order placement."""
    return {"status": "submitted", "details": kwargs}


def get_positions():
    """Stubbed positions."""
    return []


__all__ = ["connect_paper", "place_order", "get_positions"]
