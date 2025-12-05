"""
Portfolio-aware ranking and simple optimization.
Uses cvxpy when available; falls back to rule-based weights.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None


def risk_adjusted_rank(df: pd.DataFrame, return_col: str = "mu", vol_col: str = "vol") -> pd.DataFrame:
    """
    Add a risk-adjusted score = return / vol.
    """
    out = df.copy()
    if return_col in out and vol_col in out:
        vol = out[vol_col].replace(0, np.nan).fillna(out[vol_col].median() or 1e-6)
        out["risk_score"] = out[return_col] / vol
    else:
        out["risk_score"] = out.get(return_col, pd.Series(0, index=out.index))
    return out


def diversify_by_sector(
    df: pd.DataFrame, sector_col: str = "Sector", score_col: str = "risk_score", max_per_sector: int = 2, top_n: int = 20
) -> pd.DataFrame:
    """
    Pick top signals with a simple sector cap.
    """
    if df.empty:
        return df
    picked = []
    sector_counts: Dict[str, int] = {}
    for _, row in df.sort_values(score_col, ascending=False).iterrows():
        sector = str(row.get(sector_col, "Unknown"))
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        picked.append(row)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(picked) >= top_n:
            break
    return pd.DataFrame(picked)


def optimize_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    max_weight: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """
    Mean-variance optimizer maximizing information ratio.
    Returns weights and expected IR.
    """
    n = len(mu)
    if cp is None or n == 0:
        w = np.ones(n) / n if n else np.array([])
        ir = float(np.dot(w, mu) / (np.sqrt(w @ cov @ w) + 1e-6)) if n else 0.0
        return w, ir

    w = cp.Variable(n)
    ret = mu @ w
    risk = cp.quad_form(w, cov)
    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
    prob = cp.Problem(cp.Maximize(ret - 0.5 * risk), constraints)
    prob.solve(solver=cp.ECOS, verbose=False)
    if w.value is None:
        w_val = np.ones(n) / n
    else:
        w_val = np.array(w.value).flatten()
    ir = float(ret.value / (np.sqrt(risk.value) + 1e-6)) if risk.value is not None else 0.0
    return w_val, ir
