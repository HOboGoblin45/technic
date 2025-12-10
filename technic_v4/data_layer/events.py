"""
Lightweight earnings/dividend event cache loader.

Expected CSV: data_cache/events_calendar.csv with columns:
    symbol, next_earnings_date, last_earnings_date, earnings_surprise_flag, dividend_ex_date
Dates in YYYY-MM-DD.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

EVENTS_PATH = Path("data_cache/events_calendar.csv")
_CACHE: Dict[str, dict] = {}


def _parse_date(val):
    ts = pd.to_datetime(val, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return ts.normalize()


def _parse_flag(val) -> bool:
    if pd.isna(val):
        return False
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "t", "yes", "y"}
    try:
        return bool(int(val))
    except Exception:
        return bool(val)


def _load_events() -> Dict[str, dict]:
    global _CACHE
    if _CACHE:
        return _CACHE
    if not EVENTS_PATH.exists():
        _CACHE = {}
        return _CACHE
    try:
        df = pd.read_csv(EVENTS_PATH)
    except Exception:
        _CACHE = {}
        return _CACHE
    if df.empty:
        _CACHE = {}
        return _CACHE
    sym_col = next((c for c in df.columns if c.lower() in {"symbol", "ticker"}), None)
    if sym_col is None:
        _CACHE = {}
        return _CACHE
    df[sym_col] = df[sym_col].astype(str).str.strip().str.upper()
    # Normalize date columns
    for c in ["next_earnings_date", "last_earnings_date", "dividend_ex_date"]:
        if c in df.columns:
            df[c] = df[c].apply(_parse_date)
    cache: Dict[str, dict] = {}
    for _, row in df.iterrows():
        sym = row[sym_col]
        cache[sym] = {
            "next_earnings_date": row.get("next_earnings_date"),
            "last_earnings_date": row.get("last_earnings_date"),
            "earnings_surprise_flag": _parse_flag(row.get("earnings_surprise_flag", False)),
            "dividend_ex_date": row.get("dividend_ex_date"),
        }
    _CACHE = cache
    return _CACHE


def get_event_info(symbol: str) -> Optional[dict]:
    if not symbol:
        return None
    cache = _load_events()
    return cache.get(symbol.upper())


__all__ = ["get_event_info", "EVENTS_PATH"]
