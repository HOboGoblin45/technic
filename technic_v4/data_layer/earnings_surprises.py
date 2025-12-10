"""
Loader for bulk earnings surprises from data_cache/earnings_surprises_bulk.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

SURPRISE_PATH = Path("data_cache/earnings_surprises_bulk.csv")
_CACHE: Optional[pd.DataFrame] = None


def load_surprises() -> pd.DataFrame:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not SURPRISE_PATH.exists():
        _CACHE = pd.DataFrame()
        return _CACHE
    try:
        df = pd.read_csv(SURPRISE_PATH)
        df["symbol"] = df["symbol"].astype(str).str.upper()
        _CACHE = df
    except Exception:
        _CACHE = pd.DataFrame()
    return _CACHE


def get_latest_surprise(symbol: str) -> Dict:
    """
    Return the most recent surprise row for a symbol (dict) or {}.
    """
    if not symbol:
        return {}
    df = load_surprises()
    if df.empty:
        return {}
    sub = df.loc[df["symbol"] == symbol.upper()]
    if sub.empty:
        return {}
    try:
        sub = sub.sort_values("date")
    except Exception:
        pass
    return sub.iloc[-1].to_dict()


__all__ = ["load_surprises", "get_latest_surprise", "SURPRISE_PATH"]
