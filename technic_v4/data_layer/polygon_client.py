from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

# Basic settings – we can tune these later or move into config
BASE_URL = "https://api.polygon.io"
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
REQUEST_TIMEOUT = 10   # seconds

session = requests.Session()


def _polygon_get(path: str, params: dict) -> Optional[requests.Response]:
    """GET wrapper with simple retry/backoff for transient errors."""
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY environment variable is not set.")

    url = f"{BASE_URL}{path}"
    params = dict(params or {})
    params["apiKey"] = api_key

    backoff = INITIAL_BACKOFF

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.Timeout:
            print(f"[POLYGON TIMEOUT] {url} (attempt {attempt})")
            if attempt == MAX_RETRIES:
                return None
            time.sleep(backoff)
            backoff *= 2
            continue

        if resp.status_code == 200:
            return resp

        # Transient / rate-limit style errors – retry with backoff
        if resp.status_code in (429, 500, 502, 503, 504):
            print(
                f"[POLYGON RETRY] {url} status={resp.status_code} attempt={attempt} "
                f"message={resp.text[:120]!r}"
            )
            if attempt == MAX_RETRIES:
                return None
            time.sleep(backoff)
            backoff *= 2
            continue

        # Other errors – don't hammer the API
        print(
            f"[POLYGON ERROR] {url} status={resp.status_code} body={resp.text[:200]!r}"
        )
        return None

    return None


def get_stock_history_df(symbol: str, days: int) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV history for `symbol` for the past `days` days.

    Returns a DataFrame indexed by date with columns:
        [Open, High, Low, Close, Volume]
    or None if data could not be retrieved.
    """
    if days <= 0:
        raise ValueError("days must be positive")

    # Compute from/to dates (Polygon wants YYYY-MM-DD)
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days + 5)  # small buffer

    path = f"/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"

    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 5000,
    }

    resp = _polygon_get(path, params)
    if resp is None:
        print(f"[POLYGON FAIL] Unable to retrieve history for {symbol}")
        return None

    try:
        data = resp.json()
    except Exception as exc:
        print(f"[POLYGON JSON ERROR] {symbol}: {exc}")
        return None

    if data.get("resultsCount", 0) == 0 or "results" not in data:
        print(f"[POLYGON NO DATA] {symbol}")
        return None

    rows = []
    for bar in data["results"]:
        ts = bar.get("t")
        if ts is None:
            continue
        # Polygon timestamps are in ms since epoch
        dt = datetime.fromtimestamp(ts / 1000, timezone.utc).date()
        rows.append(
            {
                "Date": dt,
                "Open": bar.get("o"),
                "High": bar.get("h"),
                "Low": bar.get("l"),
                "Close": bar.get("c"),
                "Volume": bar.get("v"),
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values("Date")
    df = df.set_index("Date")

    return df


def get_stock_intraday_df(
    symbol: str,
    days: int,
    multiplier: int = 5,
    timespan: str = "minute",
) -> Optional[pd.DataFrame]:
    """
    Fetch intraday OHLCV for `symbol` for the past `days` days.
    Defaults to 5-minute bars to keep responses lighter.

    Returns a DataFrame indexed by timestamp with columns:
        [Open, High, Low, Close, Volume]
    """
    if days <= 0:
        raise ValueError("days must be positive")

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days + 1)

    path = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date:%Y-%m-%d}/{end_date:%Y-%m-%d}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
    }

    resp = _polygon_get(path, params)
    if resp is None:
        print(f"[POLYGON FAIL] Unable to retrieve intraday for {symbol}")
        return None

    try:
        data = resp.json()
    except Exception as exc:
        print(f"[POLYGON JSON ERROR] {symbol} intraday: {exc}")
        return None

    if data.get("resultsCount", 0) == 0 or "results" not in data:
        print(f"[POLYGON NO DATA] {symbol} intraday")
        return None

    rows = []
    for bar in data["results"]:
        ts = bar.get("t")
        if ts is None:
            continue
        dt_ts = datetime.utcfromtimestamp(ts / 1000)
        rows.append(
            {
                "Date": dt_ts,
                "Open": bar.get("o"),
                "High": bar.get("h"),
                "Low": bar.get("l"),
                "Close": bar.get("c"),
                "Volume": bar.get("v"),
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values("Date")
    df = df.set_index("Date")
    return df
