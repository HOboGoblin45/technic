"""
Historical replay / ICS backtester.

Runs the scanner in an "as-of" mode for a range of trading days and writes a
flat table with signals, TechRating, ICS, playstyle, and forward returns.
This is deliberately lightweight so we can build depth over time.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from technic_v4 import data_engine
from technic_v4.engine.scoring import compute_scores
from technic_v4.scanner_core import _passes_basic_filters
from technic_v4.universe_loader import load_universe


DEFAULT_LOOKBACK_DAYS = 200
DEFAULT_MAX_SYMBOLS = 200
DEFAULT_HORIZONS = "5,10,21"
DEFAULT_OUT = Path("technic_v4/scanner_output/history/replay_ics.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Historical replay / ICS sampler")
    parser.add_argument("--start", type=str, default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=datetime.utcnow().date().isoformat(), help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-symbols", type=int, default=DEFAULT_MAX_SYMBOLS, help="Number of symbols to include")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS, help="Min history length before scoring")
    parser.add_argument(
        "--horizons",
        type=str,
        default=DEFAULT_HORIZONS,
        help="Comma-separated forward horizons in trading days (e.g., 5,10,21)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT),
        help="Output parquet path (default technic_v4/scanner_output/history/replay_ics.parquet)",
    )
    return parser.parse_args()


def trading_dates(start: datetime, end: datetime, index_dates: Optional[pd.DatetimeIndex] = None) -> List[pd.Timestamp]:
    """
    Build a list of trading dates. If index_dates (e.g., SPY history) is provided,
    use that; otherwise fall back to business days.
    """
    if index_dates is not None and len(index_dates) > 0:
        mask = (index_dates >= pd.Timestamp(start)) & (index_dates <= pd.Timestamp(end))
        return list(index_dates[mask])
    return list(pd.date_range(start=start, end=end, freq="B"))


def _classify_playstyle_row(row: pd.Series) -> str:
    """A lightweight playstyle classifier using risk_score."""
    risk_val = row.get("risk_score", np.nan)
    try:
        if pd.notna(risk_val):
            if risk_val >= 0.2:
                return "Stable"
            if risk_val < 0.12:
                return "Explosive"
    except Exception:
        pass
    return "Neutral"


def compute_forward_returns(df_prices: pd.DataFrame, anchor_date: pd.Timestamp, horizons: Iterable[int]) -> Dict[str, float]:
    """
    Compute forward returns from anchor_date for each horizon (in trading days).
    Returns a dict like {"fwd_ret_5d": 0.02, ...}
    """
    out: Dict[str, float] = {}
    if df_prices is None or df_prices.empty:
        for h in horizons:
            out[f"fwd_ret_{h}d"] = np.nan
        return out

    if not isinstance(df_prices.index, pd.DatetimeIndex):
        df_prices = df_prices.copy()
        df_prices.index = pd.to_datetime(df_prices.index)
    df_prices = df_prices.sort_index()

    if anchor_date not in df_prices.index:
        # if anchor_date missing, take last available before anchor
        df_prices = df_prices[df_prices.index <= anchor_date]
    if df_prices.empty:
        for h in horizons:
            out[f"fwd_ret_{h}d"] = np.nan
        return out

    try:
        anchor_close = float(df_prices.loc[df_prices.index.max(), "Close"])
    except Exception:
        anchor_close = np.nan

    for h in horizons:
        target_date = anchor_date + timedelta(days=h)
        fut = df_prices[df_prices.index >= target_date]
        if fut.empty or np.isnan(anchor_close):
            out[f"fwd_ret_{h}d"] = np.nan
        else:
            fut_close = float(fut.iloc[0]["Close"])
            out[f"fwd_ret_{h}d"] = (fut_close - anchor_close) / anchor_close
    return out


def main() -> None:
    args = parse_args()
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)

    symbols = [u.symbol for u in load_universe()][: args.max_symbols]
    if not symbols:
        print("No symbols found in universe loader.")
        return

    # Try to get SPY history for trading calendar
    spy_hist = data_engine.get_price_history("SPY", days=4000, freq="daily")
    spy_hist = spy_hist.sort_index() if not spy_hist.empty else pd.DataFrame()
    index_dates = spy_hist.index if isinstance(spy_hist.index, pd.DatetimeIndex) else None
    trade_days = trading_dates(start_dt, end_dt, index_dates=index_dates)

    max_days_needed = args.lookback_days + max(horizons) + 30
    price_cache: Dict[str, pd.DataFrame] = {}
    rows: List[Dict[str, object]] = []

    for sym in symbols:
        try:
            hist = data_engine.get_price_history(sym, days=max_days_needed + 500, freq="daily")
        except Exception:
            hist = pd.DataFrame()
        if hist is None or hist.empty:
            continue
        hist = hist.sort_index()
        if not isinstance(hist.index, pd.DatetimeIndex):
            hist.index = pd.to_datetime(hist.index)
        price_cache[sym] = hist

    for as_of in trade_days:
        for sym in symbols:
            df_hist = price_cache.get(sym)
            if df_hist is None or df_hist.empty:
                continue
            df_slice = df_hist[df_hist.index <= as_of].copy()
            if len(df_slice) < args.lookback_days:
                continue
            if not _passes_basic_filters(df_slice):
                continue

            scored = compute_scores(df_slice, trade_style="Short-term swing", fundamentals=None)
            if scored is None or scored.empty:
                continue
            latest = scored.iloc[-1].copy()
            latest_dict = latest.to_dict()
            latest_dict["scan_date"] = as_of.date().isoformat()
            latest_dict["Symbol"] = sym

            # DollarVolume helper
            if "Close" in df_slice.columns and "Volume" in df_slice.columns:
                latest_dict["DollarVolume"] = float(df_slice["Close"].iloc[-1] * df_slice["Volume"].iloc[-1])

            # PlayStyle / stability flags
            ps = _classify_playstyle_row(latest)
            latest_dict["PlayStyle"] = ps
            latest_dict["IsStable"] = ps == "Stable"
            latest_dict["IsHighRisk"] = ps == "Explosive"

            # Basic ICS approximation (local, not cross-sectional)
            tech_val = float(latest_dict.get("TechRating", 0.0) or 0.0)
            tech_term = min(max(tech_val / 30.0, 0.0), 1.0)
            alpha_term = float(latest_dict.get("AlphaScore", 0.0) or 0.0)
            alpha_term = max(min(alpha_term, 1.0), 0.0) if not np.isnan(alpha_term) else 0.0
            liquidity_term = 0.0
            dv = latest_dict.get("DollarVolume")
            if dv is not None and not pd.isna(dv):
                liquidity_term = min(float(dv) / 20_000_000.0, 1.0)
            stability_term = 1.0 if ps == "Stable" else 0.0

            ics = (
                0.40 * tech_term
                + 0.25 * alpha_term
                + 0.15 * 0.0  # sector-neutral alpha not available here
                + 0.10 * stability_term
                + 0.10 * liquidity_term
            )
            latest_dict["InstitutionalCoreScore"] = round(ics * 100.0, 2)

            # Forward returns
            fwd = compute_forward_returns(df_hist, df_slice.index.max(), horizons)
            latest_dict.update(fwd)

            rows.append(latest_dict)

    if not rows:
        print("No rows produced; check inputs.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
