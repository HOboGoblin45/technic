"""
Loader for quality scores cache.
Prefers a locally built quality_scores.csv; if missing, derives from bulk
financial data in data_cache/*_bulk.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

QUALITY_PATH = Path("data_cache/quality_scores.csv")
FINANCIAL_SCORES_BULK = Path("data_cache/financial_scores_bulk.csv")
RATIOS_TTM_BULK = Path("data_cache/ratios_ttm_bulk.csv")
_CACHE: Optional[pd.DataFrame] = None


def _build_from_bulk() -> pd.DataFrame:
    if not FINANCIAL_SCORES_BULK.exists():
        return pd.DataFrame()

    df_fs = pd.read_csv(FINANCIAL_SCORES_BULK)
    df_fs["symbol"] = df_fs["symbol"].astype(str).str.upper()

    # Optional margins from ratios
    margins = pd.DataFrame()
    if RATIOS_TTM_BULK.exists():
        try:
            margins = pd.read_csv(RATIOS_TTM_BULK)
            margins["symbol"] = margins["symbol"].astype(str).str.upper()
            margins = margins[
                ["symbol", "netProfitMarginTTM", "grossProfitMarginTTM", "ebitdaMarginTTM"]
            ]
        except Exception:
            margins = pd.DataFrame()

    df = df_fs.merge(margins, on="symbol", how="left")

    # Compute a simple 0-100 quality score from Piotroski (0-9 scale)
    def _scale_pio(val):
        try:
            return float(val) / 9.0 * 100.0
        except Exception:
            return None

    df["QualityScore"] = df["piotroskiScore"].apply(_scale_pio)
    df = df.rename(columns={"piotroskiScore": "piotroski_score", "altmanZScore": "altman_z"})

    # Keep compact set of columns
    keep_cols = [
        "symbol",
        "QualityScore",
        "piotroski_score",
        "altman_z",
        "netProfitMarginTTM",
        "grossProfitMarginTTM",
        "ebitdaMarginTTM",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Persist for future loads
    try:
        df.to_csv(QUALITY_PATH, index=False)
    except Exception:
        pass

    return df


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
