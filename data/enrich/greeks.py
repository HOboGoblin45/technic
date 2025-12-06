"""Option Greeks enrichment from Polygon."""

from __future__ import annotations

from polygon import RESTClient


def fetch_greeks(api_key: str, underlying: str):
    client = RESTClient(api_key)
    # Placeholder: polygon greeks endpoint usage; stub for now.
    return []


__all__ = ["fetch_greeks"]
