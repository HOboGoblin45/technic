"""Sector risk weighting engine."""

from __future__ import annotations


def compute_sector_risk_weights(sector_vol_dict):
    """Assign portfolio-level risk weights based on sector vol."""
    total_vol = sum(sector_vol_dict.values())
    if total_vol == 0:
        return {sector: 0 for sector in sector_vol_dict}
    return {sector: vol / total_vol for sector, vol in sector_vol_dict.items()}


__all__ = ["compute_sector_risk_weights"]
