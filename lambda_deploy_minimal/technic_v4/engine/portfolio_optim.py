"""
Portfolio & risk weighting utilities:
 - Mean-variance style weights (with sector caps)
 - Inverse-volatility / risk-parity weights
 - HRP-style (sector-clustered) weights using a simple correlation proxy
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _build_corr_proxy(
    vols: pd.Series,
    sector: Optional[pd.Series] = None,
    intra_corr: float = 0.6,
    inter_corr: float = 0.2,
) -> pd.DataFrame:
    """
    Construct a simple correlation proxy:
      - within-sector correlation = intra_corr
      - cross-sector correlation = inter_corr
    """
    idx = vols.index
    if sector is None:
        corr = pd.DataFrame(inter_corr, index=idx, columns=idx)
    else:
        sector = sector.fillna("UNKNOWN")
        corr = pd.DataFrame(inter_corr, index=idx, columns=idx)
        for sec, sec_idx in sector.groupby(sector).groups.items():
            corr.loc[sec_idx, sec_idx] = intra_corr
    np.fill_diagonal(corr.values, 1.0)
    return corr


def _cov_from_vols(vols: pd.Series, sector: Optional[pd.Series] = None) -> pd.DataFrame:
    """Build covariance matrix from vols and a simple correlation proxy."""
    vols = vols.astype(float).replace(0, np.nan).fillna(vols.median() or 0.02)
    corr = _build_corr_proxy(vols, sector)
    cov = corr.values * np.outer(vols, vols)
    return pd.DataFrame(cov, index=vols.index, columns=vols.index)


def inverse_variance_weights(
    vols: pd.Series, sector: Optional[pd.Series] = None, sector_cap: float = 0.3
) -> pd.Series:
    """Inverse-volatility weights with optional sector caps."""
    vols = vols.astype(float).replace(0, np.nan)
    inv = 1.0 / vols
    inv = inv.replace([np.inf, -np.inf], np.nan).fillna(0)
    if inv.sum() == 0:
        inv = pd.Series(1.0, index=inv.index)
    if sector is not None:
        total_raw = inv.sum()
        sector_sum = inv.groupby(sector).transform("sum")
        cap_val = sector_cap * total_raw
        scale = np.where(sector_sum > 0, np.minimum(1.0, cap_val / sector_sum), 1.0)
        inv = inv * scale
    return inv / inv.sum()


def mean_variance_weights(
    df: pd.DataFrame,
    ret_col: str = "MuTotal",
    vol_col: str = "ATR14_pct",
    sector_col: str = "Sector",
    sector_cap: float = 0.3,
    risk_aversion: float = 10.0,
) -> pd.Series:
    """
    Long-only mean-variance-ish weights using a correlation proxy (no shorting).
    If covariance inversion fails, falls back to Sharpe-style diagonal weighting.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    rets = pd.to_numeric(df.get(ret_col, 0.0), errors="coerce").fillna(0.0).clip(lower=0)
    vols = pd.to_numeric(df.get(vol_col, 0.02), errors="coerce").replace(0, 0.02)
    sector = df.get(sector_col)
    cov = _cov_from_vols(vols, sector)
    try:
        cov_mat = cov.values
        inv_cov = np.linalg.pinv(cov_mat)
        mu = rets.values.reshape(-1, 1)
        # classical MV: weights roughly proportional to inv_cov * mu (risk_aversion scales aggressiveness)
        raw = (inv_cov @ mu).flatten()
        raw = np.maximum(raw, 0)
        if raw.sum() == 0:
            raw = rets / vols
        raw = pd.Series(raw, index=df.index)
    except Exception:
        raw = rets / vols
        raw = raw.replace([np.inf, -np.inf], 0).fillna(0)
        if raw.sum() == 0:
            raw = pd.Series(1.0, index=df.index)

    if sector is not None:
        total_raw = raw.sum()
        sector_sum = raw.groupby(sector).transform("sum")
        cap_val = sector_cap * total_raw
        scale = np.where(sector_sum > 0, np.minimum(1.0, cap_val / sector_sum), 1.0)
        raw = raw * scale

    weights = raw / raw.sum()
    return weights


def _cluster_inverse_var(cov: pd.DataFrame, sectors: Optional[pd.Series]) -> Tuple[pd.Series, pd.Series]:
    """
    Compute inverse-variance weights at (a) per-sector cluster level and (b) within-sector.
    Returns: (cluster_weight, leaf_weight)
    """
    vols = np.sqrt(np.diag(cov.values))
    vols_s = pd.Series(vols, index=cov.index)
    if sectors is None:
        cluster_w = pd.Series(1.0, index=["ALL"])
        leaf_w = inverse_variance_weights(vols_s)
        return cluster_w, leaf_w
    sectors = sectors.fillna("UNKNOWN")
    leaf_w = inverse_variance_weights(vols_s, sectors)
    cluster_var = {}
    for sec, idx in sectors.groupby(sectors).groups.items():
        sub_cov = cov.loc[idx, idx]
        # portfolio variance within sector using inverse-var weights
        w = leaf_w.loc[idx].values
        var = float(w.T @ sub_cov.values @ w)
        cluster_var[sec] = var
    cluster_inv_var = pd.Series(cluster_var).replace(0, np.nan)
    cluster_inv_var = 1.0 / np.sqrt(cluster_inv_var)
    cluster_inv_var = cluster_inv_var.fillna(0)
    if cluster_inv_var.sum() == 0:
        cluster_inv_var[:] = 1.0
    cluster_w = cluster_inv_var / cluster_inv_var.sum()
    return cluster_w, leaf_w


def hrp_weights(df: pd.DataFrame, vol_col: str = "ATR14_pct", sector_col: str = "Sector", sector_cap: float = 0.3) -> pd.Series:
    """
    Simple HRP-style weights:
      - Build covariance from vols + sector correlation proxy
      - Inverse variance within sector
      - Inverse variance across sector clusters
      - Apply sector cap
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    vols = pd.to_numeric(df.get(vol_col, 0.02), errors="coerce").replace(0, 0.02)
    sectors = df.get(sector_col)
    cov = _cov_from_vols(vols, sectors)
    cluster_w, leaf_w = _cluster_inverse_var(cov, sectors)
    weights = pd.Series(0.0, index=df.index)
    if sectors is None:
        weights = leaf_w
    else:
        for sec, idx in sectors.fillna("UNKNOWN").groupby(sectors.fillna("UNKNOWN")).groups.items():
            weights.loc[idx] = leaf_w.loc[idx] * cluster_w.get(sec, 0.0)
    # Sector cap pass
    if sectors is not None:
        total_raw = weights.sum()
        sector_sum = weights.groupby(sectors.fillna("UNKNOWN")).transform("sum")
        cap_val = sector_cap * total_raw
        scale = np.where(sector_sum > 0, np.minimum(1.0, cap_val / sector_sum), 1.0)
        weights = weights * scale
    if weights.sum() > 0:
        weights = weights / weights.sum()
    return weights
