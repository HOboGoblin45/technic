"""
Loader for quality scores cache at data_cache/quality_scores.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

QUALITY_PATH = Path("data_cache/quality_scores.csv")
_CACHE: Optional[pd.DataFrame] = None


def load_quality() -> pd.DataFrame:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not QUALITY_PATH.exists():
        _CACHE = pd.DataFrame()
        return _CACHE
    try:
        df = pd.read_csv(QUALITY_PATH)
        df["symbol"] = df["symbol"].astype(str).str.upper()
        _CACHE = df
    except Exception:
        _CACHE = pd.DataFrame()
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
