"""
Loader for ratings/price targets cache at data_cache/ratings_targets.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

RATINGS_PATH = Path("data_cache/ratings_targets.csv")
_CACHE: Optional[pd.DataFrame] = None


def load_ratings() -> pd.DataFrame:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not RATINGS_PATH.exists():
        _CACHE = pd.DataFrame()
        return _CACHE
    try:
        df = pd.read_csv(RATINGS_PATH)
        df["symbol"] = df["symbol"].astype(str).str.upper()
        _CACHE = df
    except Exception:
        _CACHE = pd.DataFrame()
    return _CACHE


def get_rating_info(symbol: str) -> Dict:
    if not symbol:
        return {}
    df = load_ratings()
    if df.empty:
        return {}
    sub = df.loc[df["symbol"] == symbol.upper()]
    if sub.empty:
        return {}
    row = sub.iloc[0].to_dict()
    return row


__all__ = ["get_rating_info", "load_ratings", "RATINGS_PATH"]
