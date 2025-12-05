# technic_v4/data_layer/bulk_daily.py

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from technic_v4.data_layer.polygon_client import _polygon_get

# Base cache directory for bulk daily snapshots
ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data_cache" / "daily"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_date(d: Union[str, date, datetime, None]) -> str:
    """
    Convert a variety of date inputs into the YYYY-MM-DD string Polygon expects.
    If d is None, use today's UTC date.
    """
    if d is None:
        d = datetime.utcnow().date()
    if isinstance(d, datetime):
        d = d.date()
    if isinstance(d, date):
        return d.strftime("%Y-%m-%d")
    if isinstance(d, str):
        # Assume user is giving a proper YYYY-MM-DD string
        return d
    raise ValueError(f"Unsupported date value: {d!r}")


def get_daily_snapshot(
    d: Union[str, date, datetime, None] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch a full-market daily snapshot (all US stocks) for a given date
    using Polygon's grouped aggregates endpoint, with on-disk caching.

    Returns a DataFrame with one row per symbol and columns:

        Symbol
        Date
        Open
        High
        Low
        Close
        Volume
        VWAP

    Notes:
    - Results are cached under data_cache/daily/daily_YYYY-MM-DD.parquet
    - If a cached file exists and force_refresh=False, it is loaded instead of
      calling Polygon again.
    """
    date_str = _normalize_date(d)
    cache_path = CACHE_DIR / f"daily_{date_str}.parquet"

    # --- Load from cache if available ---
    if cache_path.exists() and not force_refresh:
        df_cached = pd.read_parquet(cache_path)
        return df_cached

    # --- Otherwise fetch from Polygon ---
    path = f"/v2/aggs/grouped/locale/us/market/stocks/{date_str}"

    params = {
        "adjusted": "true",
        # If you want OTC as well, set include_otc=true. For now we keep it false.
        "include_otc": "false",
    }

    resp = _polygon_get(path, params)
    if resp is None:
        raise RuntimeError(f"Unable to fetch grouped daily aggregates for {date_str}")

    try:
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Failed to parse Polygon grouped JSON for {date_str}: {exc}") from exc

    results = data.get("results", [])
    if not results:
        raise RuntimeError(f"No grouped daily data returned for {date_str}")

    rows = []
    for item in results:
        symbol = item.get("T")
        if not symbol:
            continue

        rows.append(
            {
                "Symbol": symbol,
                "Date": date_str,
                "Open": item.get("o"),
                "High": item.get("h"),
                "Low": item.get("l"),
                "Close": item.get("c"),
                "Volume": item.get("v"),
                "VWAP": item.get("vw"),
            }
        )

    if not rows:
        raise RuntimeError(f"Grouped daily data for {date_str} contained no valid rows.")

    df = pd.DataFrame(rows)
    # Optional: set Date as a datetime column
    df["Date"] = pd.to_datetime(df["Date"])

    # Save to parquet cache for fast reload
    df.to_parquet(cache_path, index=False)

    return df
