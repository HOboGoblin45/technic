"""
Loader for sponsorship/insider signals.
Consumes:
  - data_cache/etf_holder_bulk.csv
  - data_cache/institutional_ownership_latest.csv
  - data_cache/insider_trading_latest.csv
Outputs per-symbol fields:
  - etf_holder_count
  - inst_holder_count
  - SponsorshipScore (pct-rank of combined holders)
  - has_recent_insider_buy (last 90d)
  - has_heavy_insider_sell (last 90d, large sells)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

ETF_PATH = Path("data_cache/etf_holder_bulk.csv")
INST_PATH = Path("data_cache/institutional_ownership_latest.csv")
INSIDER_PATH = Path("data_cache/insider_trading_latest.csv")
ETF_HOLDINGS_PATH = Path("technic_v4/data_cache/etf_holdings.parquet")
SPONSOR_THIN_PATH = Path("data_cache/sponsorship_cache.csv")

_SPONSOR_CACHE: Optional[pd.DataFrame] = None
_INSIDER_CACHE: Optional[pd.DataFrame] = None


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df["symbol"] = df["symbol"].astype(str).str.upper()
        return df
    except Exception:
        return pd.DataFrame()


def _thin_to_universe(etf: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    """
    Filter heavy bulk files down to the Technic universe and persist a slim cache.
    """
    try:
        from technic_v4.universe_loader import load_universe
    except Exception:
        return pd.DataFrame()

    try:
        universe = load_universe()
        symbols = {row.symbol.upper() for row in universe}
    except Exception:
        symbols = set()

    if not symbols:
        return pd.DataFrame()

    if not etf.empty:
        etf = etf[etf["symbol"].isin(symbols)]
    if not inst.empty:
        inst = inst[inst["symbol"].isin(symbols)]

    etf_counts = etf.groupby("symbol").size().rename("etf_holder_count") if not etf.empty else pd.Series(dtype=int)
    inst_counts = inst.groupby("symbol").size().rename("inst_holder_count") if not inst.empty else pd.Series(dtype=int)

    # Aggregate ETF holdings parquet if present (stock-level exposure)
    etf_h_agg = pd.DataFrame()
    if ETF_HOLDINGS_PATH.exists():
        try:
            etf_h = pd.read_parquet(ETF_HOLDINGS_PATH)
            etf_h["asset_symbol"] = etf_h["asset_symbol"].astype(str).str.upper()
            etf_h_agg = etf_h.groupby("asset_symbol").agg(
                etf_weight_sum_pct=pd.NamedAgg(column="weight_pct", aggfunc="sum"),
                etf_holder_count_etf=pd.NamedAgg(column="etf_symbol", aggfunc="nunique"),
            )
            etf_h_agg.index.name = "symbol"
        except Exception:
            etf_h_agg = pd.DataFrame()

    df = pd.DataFrame(index=etf_counts.index.union(inst_counts.index).union(etf_h_agg.index))
    if not etf_counts.empty:
        df = df.join(etf_counts, how="left")
    if not inst_counts.empty:
        df = df.join(inst_counts, how="left")
    if not etf_h_agg.empty:
        df = df.join(etf_h_agg, how="left")

    df = df.fillna(0)
    for col in ("etf_holder_count", "inst_holder_count", "etf_holder_count_etf", "etf_weight_sum_pct"):
        if col not in df:
            df[col] = 0
    combined = df["etf_holder_count"] + df["inst_holder_count"] + df["etf_holder_count_etf"]
    df["SponsorshipScore"] = combined.rank(pct=True).astype(float) * 100.0
    thin = df.reset_index().rename(columns={"index": "symbol"})

    try:
        thin.to_csv(SPONSOR_THIN_PATH, index=False)
    except Exception:
        pass
    return thin


def load_sponsorship() -> pd.DataFrame:
    global _SPONSOR_CACHE
    if _SPONSOR_CACHE is not None:
        return _SPONSOR_CACHE

    # Prefer slim cache if present
    if SPONSOR_THIN_PATH.exists():
        try:
            slim = pd.read_csv(SPONSOR_THIN_PATH)
            slim["symbol"] = slim["symbol"].astype(str).str.upper()
            # If cache lacks ETF-weight info or is tiny, rebuild
            if ("etf_weight_sum_pct" in slim.columns) and len(slim) > 50:
                _SPONSOR_CACHE = slim
                return _SPONSOR_CACHE
        except Exception:
            pass

    etf = _load_csv(ETF_PATH)
    inst = _load_csv(INST_PATH)

    slim = _thin_to_universe(etf, inst)
    if not slim.empty:
        _SPONSOR_CACHE = slim
        return _SPONSOR_CACHE

    # Fallback: compute across full data if universe unavailable
    etf_counts = etf.groupby("symbol").size().rename("etf_holder_count") if not etf.empty else pd.Series(dtype=int)
    inst_counts = inst.groupby("symbol").size().rename("inst_holder_count") if not inst.empty else pd.Series(dtype=int)

    df = pd.DataFrame(index=etf_counts.index.union(inst_counts.index))
    if not etf_counts.empty:
        df = df.join(etf_counts, how="left")
    if not inst_counts.empty:
        df = df.join(inst_counts, how="left")

    df = df.fillna(0)
    combined = df["etf_holder_count"] + df["inst_holder_count"]
    df["SponsorshipScore"] = combined.rank(pct=True).astype(float) * 100.0

    _SPONSOR_CACHE = df.reset_index().rename(columns={"index": "symbol"})
    return _SPONSOR_CACHE


def load_insiders() -> pd.DataFrame:
    global _INSIDER_CACHE
    if _INSIDER_CACHE is not None:
        return _INSIDER_CACHE
    df = _load_csv(INSIDER_PATH)
    if df.empty:
        _INSIDER_CACHE = df
        return df
    if "transactionDate" in df.columns:
        df["transactionDate"] = pd.to_datetime(df["transactionDate"], errors="coerce")
    _INSIDER_CACHE = df
    return _INSIDER_CACHE


def get_sponsorship(symbol: str) -> Dict:
    if not symbol:
        return {}
    df = load_sponsorship()
    if df.empty:
        return {}
    sub = df.loc[df["symbol"] == symbol.upper()]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


def get_insider_flags(symbol: str, days: int = 90) -> Dict:
    if not symbol:
        return {}
    df = load_insiders()
    if df.empty:
        return {}
    sym = symbol.upper()
    sub = df.loc[df["symbol"] == sym]
    if sub.empty or "transactionDate" not in sub.columns:
        return {}
    cutoff = datetime.utcnow() - timedelta(days=days)
    recent = sub.loc[sub["transactionDate"] >= cutoff]
    if recent.empty:
        return {}
    action_col = None
    for c in recent.columns:
        if c.lower() in ("transactiontype", "type", "transaction"):
            action_col = c
            break
    if action_col is None:
        return {}
    actions = recent[action_col].astype(str).str.lower()
    has_buy = actions.str.contains("buy").any()
    # heavy sell: multiple sells or any sell with large dollar value if column exists
    has_sell = actions.str.contains("sell").any()
    has_heavy_sell = False
    if has_sell:
        if "totalValue" in recent.columns:
            vals = pd.to_numeric(recent["totalValue"], errors="coerce")
            has_heavy_sell = bool((vals > 1_000_000).any())
        else:
            has_heavy_sell = len(recent.loc[actions.str.contains("sell")]) >= 3

    return {
        "has_recent_insider_buy": bool(has_buy),
        "has_heavy_insider_sell": bool(has_heavy_sell),
    }


__all__ = ["load_sponsorship", "get_sponsorship", "load_insiders", "get_insider_flags"]
