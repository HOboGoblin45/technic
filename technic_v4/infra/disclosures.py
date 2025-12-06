"""Risk disclosure engine."""

from __future__ import annotations


def generate_disclosure(vol_band: str, drift: bool, confidence_decay: bool) -> str:
    """Generate formatted disclosures based on risk conditions."""
    parts = [f"Volatility: {vol_band}."]
    if drift:
        parts.append("Data drift detected; results may deviate from historical patterns.")
    if confidence_decay:
        parts.append("Model confidence has decayed; signals may be less reliable.")
    parts.append("This is not investment advice; use at your own risk.")
    return " ".join(parts)


__all__ = ["generate_disclosure"]
