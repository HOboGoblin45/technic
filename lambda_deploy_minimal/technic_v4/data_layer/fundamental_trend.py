"""
Loader for fundamental momentum using bulk growth CSVs.
Consumes:
  - data_cache/income_statement_growth_bulk.csv
  - data_cache/cash_flow_statement_growth_bulk.csv
Builds a per-symbol trend snapshot with revenue/net income/margin growth and a composite score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

INC_GROWTH_PATH = Path("data_cache/income_statement_growth_bulk.csv")
CF_GROWTH_PATH = Path("data_cache/cash_flow_statement_growth_bulk.csv")
_CACHE: Optional[pd.DataFrame] = None


def _load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df["symbol"] = df["symbol"].astype(str).str.upper()
        return df
    except Exception:
        return pd.DataFrame()


def load_fundamental_trend() -> pd.DataFrame:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    inc = _load_df(INC_GROWTH_PATH)
    cf = _load_df(CF_GROWTH_PATH)

    if inc.empty and cf.empty:
        _CACHE = pd.DataFrame()
        return _CACHE

    # pick latest per symbol by date
    def latest(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.sort_values("date")
        return df.groupby("symbol").tail(1)

    inc_latest = latest(inc)
    cf_latest = latest(cf)

    df = inc_latest.merge(cf_latest, on="symbol", how="outer", suffixes=("_inc", "_cf"))

    rev = pd.to_numeric(df.get("growthRevenue"), errors="coerce")
    margin = pd.to_numeric(df.get("growthGrossProfitRatio"), errors="coerce")
    net_inc = pd.to_numeric(df.get("growthNetIncome"), errors="coerce")
    eps = pd.to_numeric(df.get("growthEPS"), errors="coerce")
    ebitda = pd.to_numeric(df.get("growthEBITDA"), errors="coerce")
    op_inc = pd.to_numeric(df.get("growthOperatingIncome"), errors="coerce")

    components = []
    for comp in (rev, margin, net_inc, eps, ebitda, op_inc):
        if comp is not None:
            comp_series = pd.Series(comp, index=df.index)
            comp_series = comp_series.clip(-1.0, 2.0) * 100.0
            components.append(comp_series)

    if components:
        comp_df = pd.concat(components, axis=1)
        df["fundamental_trend_score"] = comp_df.mean(axis=1, skipna=True)
    else:
        df["fundamental_trend_score"] = np.nan

    cols = [
        "symbol",
        "fundamental_trend_score",
        "growthRevenue",
        "growthNetIncome",
        "growthEPS",
        "growthEBITDA",
        "growthOperatingIncome",
        "growthGrossProfitRatio",
    ]
    keep = [c for c in cols if c in df.columns]
    df_out = df[keep].copy()

    _CACHE = df_out
    return _CACHE


def get_fundamental_trend(symbol: str) -> Dict:
    if not symbol:
        return {}
    df = load_fundamental_trend()
    if df.empty:
        return {}
    sub = df.loc[df["symbol"] == symbol.upper()]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


__all__ = ["load_fundamental_trend", "get_fundamental_trend"]
