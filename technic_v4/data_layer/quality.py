"""
Loader for quality scores cache.
Prefers a locally built quality_scores.csv; if missing, derives from bulk
financial data in data_cache/*_bulk.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

QUALITY_PATH = Path("data_cache/quality_scores.csv")
FINANCIAL_SCORES_BULK = Path("data_cache/financial_scores_bulk.csv")
RATIOS_TTM_BULK = Path("data_cache/ratios_ttm_bulk.csv")
KEY_METRICS_TTM_BULK = Path("data_cache/key_metrics_ttm_bulk.csv")
_CACHE: Optional[pd.DataFrame] = None


def _sector_map() -> Dict[str, str]:
    try:
        from technic_v4.universe_loader import load_universe
    except Exception:
        return {}
    try:
        rows = load_universe()
    except Exception:
        return {}
    return {r.symbol.upper(): (r.sector or "UNKNOWN") for r in rows}


def _pct_rank(series: pd.Series, invert: bool = False) -> pd.Series:
    if series.empty:
        return series
    s = series.copy()
    if invert:
        s = -s
    return s.rank(pct=True).astype(float)


def _build_from_bulk() -> pd.DataFrame:
    if not FINANCIAL_SCORES_BULK.exists() or not KEY_METRICS_TTM_BULK.exists() or not RATIOS_TTM_BULK.exists():
        return pd.DataFrame()

    df_fs = pd.read_csv(FINANCIAL_SCORES_BULK)
    df_fs["symbol"] = df_fs["symbol"].astype(str).str.upper()

    df_rat = pd.read_csv(RATIOS_TTM_BULK)
    df_rat["symbol"] = df_rat["symbol"].astype(str).str.upper()

    df_km = pd.read_csv(KEY_METRICS_TTM_BULK)
    df_km["symbol"] = df_km["symbol"].astype(str).str.upper()

    df = df_fs.merge(df_rat, on="symbol", how="left").merge(df_km, on="symbol", how="left", suffixes=("", "_km"))

    # Factors
    factors = {
        "roe": df.get("returnOnEquityTTM"),
        "roa": df.get("returnOnAssetsTTM"),
        "gross_margin": df.get("grossProfitMarginTTM"),
        "operating_margin": df.get("operatingProfitMarginTTM"),
        "net_margin": df.get("netProfitMarginTTM"),
        "fcf_yield": df.get("freeCashFlowYieldTTM"),
        "net_debt_ebitda": df.get("netDebtToEBITDATTM"),  # lower better
        "interest_coverage": df.get("interestCoverageRatioTTM"),
    }

    sector_map = _sector_map()
    df["sector"] = df["symbol"].map(sector_map).fillna("UNKNOWN")

    comp_scores = []
    for name, series in factors.items():
        if series is None:
            comp_scores.append(pd.Series(np.nan, index=df.index))
            continue
        by_sector = series.groupby(df["sector"])
        ranks = by_sector.transform(lambda s: _pct_rank(s, invert=(name == "net_debt_ebitda")) * 100.0)
        comp_scores.append(ranks)
        df[f"{name}_sector_pct"] = ranks

    if comp_scores:
        comp_df = pd.concat(comp_scores, axis=1)
        df["QualityScore"] = comp_df.mean(axis=1, skipna=True)

    df = df.rename(columns={"piotroskiScore": "piotroski_score", "altmanZScore": "altman_z"})

    # Persist for future loads
    out_cols = [
        "symbol",
        "sector",
        "QualityScore",
        "piotroski_score",
        "altman_z",
        "roe_sector_pct",
        "roa_sector_pct",
        "gross_margin_sector_pct",
        "operating_margin_sector_pct",
        "net_margin_sector_pct",
        "fcf_yield_sector_pct",
        "net_debt_ebitda_sector_pct",
        "interest_coverage_sector_pct",
        "netDebtToEBITDATTM",
        "interestCoverageRatioTTM",
        "freeCashFlowYieldTTM",
        "returnOnEquityTTM",
        "returnOnAssetsTTM",
        "grossProfitMarginTTM",
        "operatingProfitMarginTTM",
        "netProfitMarginTTM",
    ]
    keep = [c for c in out_cols if c in df.columns]
    df_out = df[keep].copy()
    try:
        df_out.to_csv(QUALITY_PATH, index=False)
    except Exception:
        pass

    return df_out


def load_quality() -> pd.DataFrame:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    if QUALITY_PATH.exists():
        try:
            df = pd.read_csv(QUALITY_PATH)
            df["symbol"] = df["symbol"].astype(str).str.upper()
            _CACHE = df
            return _CACHE
        except Exception:
            _CACHE = None  # fall through to rebuild

    df_built = _build_from_bulk()
    _CACHE = df_built if df_built is not None else pd.DataFrame()
    return _CACHE


def get_quality_info(symbol: str) -> Dict:
    if not symbol:
        return {}
    df = load_quality()
    if df.empty:
        return {}
    sub = df.loc[df["symbol"] == symbol.upper()]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


__all__ = ["get_quality_info", "load_quality", "QUALITY_PATH"]
