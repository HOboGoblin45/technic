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

try:
    from sklearn.covariance import LedoitWolf
    HAVE_LW = True
except Exception:  # pragma: no cover
    HAVE_LW = False


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


def estimate_covariance(df_returns: pd.DataFrame) -> np.ndarray:
    """
    Estimate covariance matrix from returns.
    Tries Ledoit-Wolf shrinkage; falls back to sample covariance.
    """
    if df_returns is None or df_returns.empty:
        return np.array([[]])
    if HAVE_LW:
        try:
            lw = LedoitWolf().fit(df_returns.fillna(0.0).values)
            return lw.covariance_
        except Exception:
            pass
    # Simple shrinkage fallback: blend sample cov with diagonal target
    sample_cov = df_returns.cov().values
    diag_target = np.diag(np.diag(sample_cov))
    shrink = 0.1
    return (1 - shrink) * sample_cov + shrink * diag_target


def build_risk_model(symbols: list[str], lookback_days: int = 60) -> dict:
    """
    Build a basic risk model: vols, covariance, and beta approximation vs SPY (if available).
    """
    from technic_v4.data_layer.price_layer import get_stock_history_df

    returns = {}
    for sym in symbols:
        try:
            hist = get_stock_history_df(symbol=sym, days=lookback_days, use_intraday=False)
            if hist is None or hist.empty or "Close" not in hist:
                continue
            rets = hist["Close"].pct_change().dropna()
            if not rets.empty:
                returns[sym] = rets
        except Exception:
            continue
    if not returns:
        return {"cov": None, "vols": None, "betas": None}
    df_returns = pd.DataFrame(returns).dropna(how="all")
    cov = estimate_covariance(df_returns)
    vols = df_returns.std()
    betas = None
    if "SPY" in df_returns.columns:
        mkt = df_returns["SPY"]
        betas = {}
        for col in df_returns.columns:
            if col == "SPY":
                continue
            cov_sm = np.cov(df_returns[col], mkt)[0][1]
            beta = cov_sm / (mkt.var() + 1e-9)
            betas[col] = beta
    return {"cov": cov, "vols": vols, "betas": betas}


def scenario_pnl(weights: pd.Series, betas: pd.Series | None, shock_pct: float = -0.05) -> float:
    """
    Approximate portfolio P&L for a uniform market shock using betas (if available).
    """
    if betas is None or betas.empty:
        return float(weights.sum() * shock_pct)
    aligned = betas.reindex(weights.index).fillna(1.0)
    return float((weights * aligned * shock_pct).sum())


def optimize_portfolio(df: pd.DataFrame, risk_settings) -> pd.DataFrame:
    """
    Portfolio optimizer: maximize sum(w * alpha) - lambda * w'Σw
    subject to sum w = 1, w >= 0, sector caps.
    Returns a DataFrame with Symbol, Weight, AlphaScore, RiskContribution.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    # Use cvxpy if available; otherwise use heuristic closed-form
    if cp is not None:
        try:
            df_local = df.copy()
            df_local["AlphaScore"] = df_local.get("AlphaScore", df_local.get("TechRating", 0)).fillna(0)
            df_local["VolatilityEstimate"] = df_local.get("VolatilityEstimate", df_local.get("vol_realized_20", 0)).fillna(0.2)
            symbols = df_local["Symbol"].tolist()
            alpha = df_local["AlphaScore"].values.astype(float)
            vols = df_local["VolatilityEstimate"].values.astype(float)

            n = len(df_local)
            w = cp.Variable(n)
            cov = df_local.at[0, "Covariance"] if "Covariance" in df_local.columns else None
            if cov is None or not isinstance(cov, np.ndarray):
                cov = np.diag(vols ** 2)

            lam = getattr(risk_settings, "risk_aversion", 0.1) if risk_settings is not None else 0.1
            sector_caps = {}
            if "Sector" in df_local.columns:
                for sector in df_local["Sector"].unique():
                    sector_caps[sector] = getattr(risk_settings, "sector_cap", 0.3) if risk_settings else 0.3

            objective = cp.Maximize(alpha @ w - lam * cp.quad_form(w, cov))
            constraints = [cp.sum(w) == 1, w >= 0]
            max_w = getattr(risk_settings, "max_weight", 0.1) if risk_settings else 0.1
            if max_w:
                constraints.append(w <= max_w)
            if sector_caps:
                for sector, cap in sector_caps.items():
                    idx = [i for i, s in enumerate(df_local["Sector"]) if s == sector]
                    if idx:
                        constraints.append(cp.sum(w[idx]) <= cap)

            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.ECOS, verbose=False)

            if w.value is not None:
                w_val = np.array(w.value).flatten()
                risk_contrib = cov @ w_val
                out = pd.DataFrame(
                    {
                        "Symbol": symbols,
                        "Weight": w_val,
                        "AlphaScore": alpha,
                        "RiskContribution": risk_contrib,
                        "VolEstimate": vols,
                        "Covariance": [cov] * len(symbols),
                    }
                )
                return out
        except Exception:
            pass

    # Heuristic fallback: closed-form w = inv(Sigma + ridge I) * mu
    df_local = df.copy()
    df_local["AlphaScore"] = df_local.get("AlphaScore", df_local.get("TechRating", 0)).fillna(0)
    df_local["VolatilityEstimate"] = df_local.get("VolatilityEstimate", df_local.get("vol_realized_20", 0)).fillna(0.2)
    symbols = df_local["Symbol"].tolist()
    alpha = df_local["AlphaScore"].values.astype(float)
    cov = df_local.at[0, "Covariance"] if "Covariance" in df_local.columns else None
    if cov is None or not isinstance(cov, np.ndarray):
        cov = np.diag(df_local["VolatilityEstimate"].values.astype(float) ** 2)
    ridge = 1e-4
    try:
        inv_cov = np.linalg.inv(cov + ridge * np.eye(len(cov)))
    except Exception:
        inv_cov = np.linalg.pinv(cov + ridge * np.eye(len(cov)))
    raw_w = inv_cov @ alpha
    raw_w = np.maximum(raw_w, 0)
    if raw_w.sum() == 0:
        raw_w = np.ones_like(raw_w)
    w_val = raw_w / raw_w.sum()
    max_w = getattr(risk_settings, "max_weight", 0.1) if risk_settings else 0.1
    if max_w:
        excess_total = 0.0
        for i, val in enumerate(w_val):
            if val > max_w:
                excess_total += val - max_w
                w_val[i] = max_w
        if excess_total > 0 and w_val.sum() > 0:
            w_val = w_val / w_val.sum()
    if "Sector" in df_local.columns:
        sector_caps = {}
        for sector in df_local["Sector"].unique():
            sector_caps[sector] = getattr(risk_settings, "sector_cap", 0.3) if risk_settings else 0.3
        for sector, cap in sector_caps.items():
            idx = [i for i, s in enumerate(df_local["Sector"]) if s == sector]
            sec_w = w_val[idx].sum()
            if sec_w > cap and sec_w > 0:
                scale = cap / sec_w
                w_val[idx] = w_val[idx] * scale
        if w_val.sum() > 0:
            w_val = w_val / w_val.sum()
    risk_contrib = cov @ w_val
    return pd.DataFrame(
        {
            "Symbol": symbols,
            "Weight": w_val,
            "AlphaScore": alpha,
            "RiskContribution": risk_contrib,
            "VolEstimate": df_local["VolatilityEstimate"].values.astype(float),
            "Covariance": [cov] * len(symbols),
        }
    )

    df_local = df.copy()
    # Required columns; fill missing with defaults
    df_local["AlphaScore"] = df_local.get("AlphaScore", df_local.get("TechRating", 0)).fillna(0)
    df_local["VolatilityEstimate"] = df_local.get("VolatilityEstimate", df_local.get("vol_realized_20", 0)).fillna(0.2)
    symbols = df_local["Symbol"].tolist()
    alpha = df_local["AlphaScore"].values.astype(float)
    vols = df_local["VolatilityEstimate"].values.astype(float)

    n = len(df_local)
    w = cp.Variable(n)
    # Diagonal covariance approximation if no matrix provided
    cov = np.diag(vols ** 2)

    lam = getattr(risk_settings, "risk_aversion", 0.1) if risk_settings is not None else 0.1
    sector_caps = {}
    if "Sector" in df_local.columns:
        for sector in df_local["Sector"].unique():
            sector_caps[sector] = 0.3

    objective = cp.Maximize(alpha @ w - lam * cp.quad_form(w, cov))
    constraints = [cp.sum(w) == 1, w >= 0]
    if sector_caps:
        for sector, cap in sector_caps.items():
            idx = [i for i, s in enumerate(df_local["Sector"]) if s == sector]
            if idx:
                constraints.append(cp.sum(w[idx]) <= cap)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except Exception:
        return pd.DataFrame()

    if w.value is None:
        return pd.DataFrame()

    w_val = np.array(w.value).flatten()
    total_var = float(w_val @ cov @ w_val)
    risk_contrib = cov @ w_val

    out = pd.DataFrame(
        {
            "Symbol": symbols,
            "Weight": w_val,
            "AlphaScore": alpha,
            "RiskContribution": risk_contrib,
            "VolEstimate": vols,
        }
    )
    return out


def apply_portfolio_weights(df: pd.DataFrame, risk_settings, use_optimizer: bool = False) -> pd.DataFrame:
    """
    Attach weights to scan results. If optimizer disabled or fails, use equal weights.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    weights = None
    if use_optimizer:
        try:
            opt = optimize_portfolio(df, risk_settings)
            if not opt.empty:
                weights = dict(zip(opt["Symbol"], opt["Weight"]))
                if "RiskContribution" in opt.columns:
                    out["RiskContribution"] = out["Symbol"].map(dict(zip(opt["Symbol"], opt["RiskContribution"])))
                if "VolEstimate" in opt.columns:
                    out["VolEstimate"] = out["Symbol"].map(dict(zip(opt["Symbol"], opt["VolEstimate"])))
        except Exception:
            weights = None
    if weights is None:
        # Inverse-volatility weighting as a simple risk-aware default
        vol_col = "VolEstimate" if "VolEstimate" in out.columns else "vol_realized_20"
        vols = out[vol_col].replace(0, np.nan) if vol_col in out.columns else pd.Series(1.0, index=out.index)
        vols = vols.fillna(vols.median() if vols.notna().any() else 1.0)
        inv_vol = 1.0 / vols
        inv_vol = inv_vol / inv_vol.sum()
        weights = dict(zip(out["Symbol"], inv_vol))
    out["Weight"] = out["Symbol"].map(weights)
    return out
