"""
Loader for bulk earnings surprises from data_cache/earnings_surprises_bulk.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

SURPRISE_PATH = Path("data_cache/earnings_surprises_bulk.csv")
_CACHE: Optional[pd.DataFrame] = None
_AGG: Optional[pd.DataFrame] = None


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["beat_flag"] = pd.to_numeric(df.get("actualEarningResult"), errors="coerce") > pd.to_numeric(
        df.get("estimatedEarning"), errors="coerce"
    )
    # percent surprise (bp)
    actual = pd.to_numeric(df.get("actualEarningResult"), errors="coerce")
    est = pd.to_numeric(df.get("estimatedEarning"), errors="coerce")
    surprise_pct = (actual - est) / est.replace(0, pd.NA)
    df["surprise_bp"] = surprise_pct * 10000.0
    return df


def _compute_agg(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    def streak(series):
        # compute current streak of beats (positive) or misses (negative)
        s = series.dropna().astype(bool).tolist()
        count = 0
        last = None
        for val in reversed(s):
            if last is None:
                last = val
            if val == last:
                count += 1
            else:
                break
        return count if last else -count

    agg_rows = []
    for sym, sub in df.groupby("symbol"):
        sub_sorted = sub.sort_values("date")
        beat_flags = sub_sorted["beat_flag"]
        streak_val = streak(beat_flags)
        last_bp = sub_sorted["surprise_bp"].iloc[-1] if "surprise_bp" in sub_sorted else None
        avg_bp = sub_sorted["surprise_bp"].mean() if "surprise_bp" in sub_sorted else None
        agg_rows.append(
            {
                "symbol": sym,
                "surprise_streak": streak_val,
                "avg_surprise_bp": avg_bp,
                "last_surprise_bp": last_bp,
                "has_beat_streak": streak_val is not None and streak_val > 0,
            }
        )
    return pd.DataFrame(agg_rows)


def load_surprises() -> pd.DataFrame:
    global _CACHE, _AGG
    if _CACHE is not None:
        return _CACHE
    if not SURPRISE_PATH.exists():
        _CACHE = pd.DataFrame()
        _AGG = pd.DataFrame()
        return _CACHE
    try:
        df = pd.read_csv(SURPRISE_PATH)
        df = _prepare(df)
        _CACHE = df
        _AGG = _compute_agg(df)
    except Exception:
        _CACHE = pd.DataFrame()
        _AGG = pd.DataFrame()
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


def get_surprise_stats(symbol: str) -> Dict:
    """
    Return aggregated surprise stats for a symbol, including streak and avg magnitude.
    """
    if not symbol:
        return {}
    _ = load_surprises()
    if _AGG is None or _AGG.empty:
        return {}
    sub = _AGG.loc[_AGG["symbol"] == symbol.upper()]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


__all__ = ["load_surprises", "get_latest_surprise", "get_surprise_stats", "SURPRISE_PATH"]
