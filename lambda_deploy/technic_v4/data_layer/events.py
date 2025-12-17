"""
Lightweight earnings/dividend event cache loader.

Expected CSV: data_cache/events_calendar.csv with columns:
    symbol, next_earnings_date, last_earnings_date, earnings_surprise_flag, dividend_ex_date
Dates in YYYY-MM-DD.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

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


def load_events_calendar() -> pd.DataFrame:
    try:
        if not EVENTS_PATH.exists():
            return pd.DataFrame()
        return pd.read_csv(EVENTS_PATH)
    except Exception:
        return pd.DataFrame()


def get_event_info(symbol: str) -> Dict[str, Any]:
    """
    Return best-effort earnings / dividend event information for a symbol.

    Fields:
      - next_earnings_date: date or None
      - last_earnings_date: date or None
      - earnings_surprise_flag: bool (True if last EPS surprise was positive)
      - dividend_ex_date: date or None

      Derived flags:
      - days_to_next_earnings: int or None
      - days_since_last_earnings: int or None
      - days_to_dividend_ex: int or None
      - is_pre_earnings_window: bool (0–5 days before earnings)
      - is_post_earnings_positive_window: bool (0–10 days after a positive surprise)
      - has_upcoming_earnings: bool
      - has_recent_positive_surprise: bool (last 30 days and positive surprise)
      - has_dividend_ex_soon: bool (0–10 days before ex‑date)
    """
    cal = load_events_calendar()
    if cal.empty:
        return {}

    sub = cal.loc[cal["symbol"].str.upper() == symbol.upper()]
    if sub.empty:
        return {}

    row = sub.iloc[0]

    def _parse_date(val: Any) -> Optional[date]:
        if pd.isna(val) or val in ("", None):
            return None
        try:
            return pd.to_datetime(str(val)).date()
        except Exception:
            return None

    next_earn = _parse_date(row.get("next_earnings_date"))
    last_earn = _parse_date(row.get("last_earnings_date"))
    div_ex = _parse_date(row.get("dividend_ex_date"))

    surprise_raw = row.get("earnings_surprise_flag", False)
    earnings_surprise_flag = bool(bool(surprise_raw) and not pd.isna(surprise_raw))

    today = date.today()

    days_to_next = (next_earn - today).days if next_earn else None
    days_since_last = (today - last_earn).days if last_earn else None
    days_to_div = (div_ex - today).days if div_ex else None

    is_pre_earnings_window = (
        days_to_next is not None and 0 <= days_to_next <= 5
    )
    is_post_earnings_positive_window = (
        earnings_surprise_flag
        and days_since_last is not None
        and 0 <= days_since_last <= 10
    )

    has_upcoming_earnings = days_to_next is not None and days_to_next >= 0
    has_recent_positive_surprise = (
        earnings_surprise_flag
        and days_since_last is not None
        and 0 <= days_since_last <= 30
    )
    has_dividend_ex_soon = (
        days_to_div is not None and 0 <= days_to_div <= 10
    )

    return {
        "next_earnings_date": next_earn,
        "last_earnings_date": last_earn,
        "earnings_surprise_flag": earnings_surprise_flag,
        "dividend_ex_date": div_ex,
        "days_to_next_earnings": days_to_next,
        "days_since_last_earnings": days_since_last,
        "days_to_dividend_ex": days_to_div,
        "is_pre_earnings_window": is_pre_earnings_window,
        "is_post_earnings_positive_window": is_post_earnings_positive_window,
        "has_upcoming_earnings": has_upcoming_earnings,
        "has_recent_positive_surprise": has_recent_positive_surprise,
        "has_dividend_ex_soon": has_dividend_ex_soon,
    }


__all__ = ["get_event_info", "EVENTS_PATH", "load_events_calendar"]
