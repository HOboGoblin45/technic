"""
Simple mean-variance portfolio optimizer (long-only, with sector caps).
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


def mean_variance_weights(
    df: pd.DataFrame,
    ret_col: str = "MuTotal",
    vol_col: str = "ATR14_pct",
    sector_col: str = "Sector",
    sector_cap: float = 0.3,
) -> pd.Series:
    """
    Compute long-only mean-variance-ish weights using a diagonal covariance proxy from vol_col.
    Sector caps enforced by scaling within sector.
    """
    if df.empty:
        return pd.Series(dtype=float)
    rets = pd.to_numeric(df.get(ret_col, 0.0), errors="coerce").fillna(0.0)
    vols = pd.to_numeric(df.get(vol_col, 0.02), errors="coerce").replace(0, 0.02)
    # Approximate Sharpe = ret/vol; use as weight seed
    seed = rets / vols
    seed = seed.clip(lower=0)
    if seed.sum() == 0:
        seed = pd.Series(1.0, index=df.index)
    # Apply sector caps
    if sector_col in df.columns:
        total_raw = seed.sum()
        sector_sum = seed.groupby(df[sector_col]).transform("sum")
        cap_val = sector_cap * total_raw
        scale = np.where(sector_sum > 0, np.minimum(1.0, cap_val / sector_sum), 1.0)
        seed = seed * scale
    weights = seed / seed.sum()
    return weights


def _calc_cov_from_returns(df: pd.DataFrame, price_cols: Optional[list[str]] = None) -> Optional[np.ndarray]:
    """
    Best-effort covariance matrix from price history columns (Close_x for each symbol).
    Expects df to have columns like ('Symbol', 'Close') repeated per symbol; for simplicity
    we skip if not available.
    """
    return None  # placeholder; full HRP would require per-symbol return matrix


def inverse_variance_weights(vols: pd.Series, sector: Optional[pd.Series] = None, sector_cap: float = 0.3) -> pd.Series:
    """Inverse-volatility weights with sector caps."""
    vols = vols.replace(0, np.nan)
    inv = 1.0 / vols
    inv = inv.fillna(0)
    if inv.sum() == 0:
        inv = pd.Series(1.0, index=inv.index)
    if sector is not None:
        total_raw = inv.sum()
        sector_sum = inv.groupby(sector).transform("sum")
        cap_val = sector_cap * total_raw
        scale = np.where(sector_sum > 0, np.minimum(1.0, cap_val / sector_sum), 1.0)
        inv = inv * scale
    return inv / inv.sum()


def hrp_weights_placeholder(vols: pd.Series, sector: Optional[pd.Series] = None, sector_cap: float = 0.3) -> pd.Series:
    """
    Placeholder for HRP-like weights; currently falls back to inverse-vol with sector cap.
    """
    return inverse_variance_weights(vols, sector, sector_cap)
