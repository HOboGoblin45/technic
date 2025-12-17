from __future__ import annotations

import json
import os
from io import StringIO
from pathlib import Path
from typing import List

import pandas as pd
import requests

from technic_v4.config.settings import get_settings


BASE_URL = "https://financialmodelingprep.com/stable/etf-holder-bulk"
DATA_DIR = Path("technic_v4/data_cache")
OUTPUT_PATH = DATA_DIR / "etf_holdings.parquet"
RAW_JSON_DIR = DATA_DIR / "etf_holdings_raw"


def _get_fmp_api_key() -> str:
    # Prefer settings, but fall back to environment for convenience
    settings = get_settings()
    api_key = (
        getattr(settings, "FMP_API_KEY", None)
        or getattr(settings, "fmp_api_key", None)
        or os.getenv("FMP_API_KEY")
    )
    if not api_key:
        raise RuntimeError("FMP_API_KEY missing in settings or environment.")
    return api_key


def fetch_bulk_part(part: int, api_key: str) -> pd.DataFrame:
    """
    Fetch one 'part' from ETF Holder Bulk API.

    Each row in the response has:
      - symbol: ETF ticker (e.g. SPY)
      - name:   ETF name
      - asset / assetSymbol: underlying asset ticker (e.g. AAPL)
      - assetName: underlying asset name
      - weightPercentage: weight of the asset in the ETF (%)
      - sharesNumber: number of shares held
      - marketValue: dollar value of this holding
      - isin, cusip: identifiers for the underlying

    The API returns a JSON array; an empty array means we've reached the end.
    """
    params = {"part": part, "apikey": api_key}
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    try:
        data = resp.json()
    except Exception:
        # Fallback to CSV parsing if JSON decode fails
        try:
            return pd.read_csv(StringIO(resp.text))
        except Exception:
            raise

    if not data:
        return pd.DataFrame()

    # Optionally persist raw JSON for debugging / inspection
    RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_JSON_DIR / f"etf_holder_part_{part}.json"
    raw_path.write_text(json.dumps(data, indent=2))

    return pd.DataFrame(data)


def normalize_etf_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names so Technic can rely on a stable schema.

    We expect FMP to provide at least:
      - 'symbol' (ETF ticker)
      - 'asset' or 'assetSymbol' (underlying ticker)
      - 'assetName'
      - 'weightPercentage'
      - 'sharesNumber'
      - 'marketValue'
      - 'isin', 'cusip'
    """
    if df.empty:
        return df

    # Pick the actual underlying symbol column used by FMP
    asset_col = None
    for cand in ("asset", "assetSymbol", "ticker"):
        if cand in df.columns:
            asset_col = cand
            break
    if asset_col is None:
        raise KeyError(
            f"Could not find an underlying asset symbol column in ETF bulk df; "
            f"available columns: {list(df.columns)}"
        )

    # Build a normalized view
    renamed = df.rename(
        columns={
            "symbol": "etf_symbol",
            asset_col: "asset_symbol",
            "assetName": "asset_name",
            "weightPercentage": "weight_pct",
            "sharesNumber": "shares",
            "marketValue": "market_value",
        }
    )

    # Keep a focused set of columns but preserve extras for debugging
    base_cols: List[str] = [
        "etf_symbol",
        "asset_symbol",
        "asset_name",
        "weight_pct",
        "shares",
        "market_value",
    ]
    optional_cols = [c for c in ("isin", "cusip", "name") if c in renamed.columns]

    cols = [c for c in base_cols + optional_cols if c in renamed.columns]
    out = renamed[cols].copy()

    # Basic cleaning
    out["etf_symbol"] = out["etf_symbol"].astype(str).str.upper()
    out["asset_symbol"] = out["asset_symbol"].astype(str).str.upper()

    return out


def build_etf_holdings_cache(max_parts: int = 10) -> None:
    """
    Download all ETF holdings via ETF Holder Bulk and write a single
    normalized parquet file to data_cache/etf_holdings.parquet.

    We iterate parts 1..max_parts until we hit an empty response.
    """
    api_key = _get_fmp_api_key()
    frames: List[pd.DataFrame] = []

    for part in range(1, max_parts + 1):
        print(f"[ETF_BULK] Fetching part {part}...")
        try:
            df_part = fetch_bulk_part(part, api_key=api_key)
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            print(f"[ETF_BULK] HTTP {status} on part {part}; stopping.")
            break
        if df_part.empty:
            print(f"[ETF_BULK] No data for part {part}; stopping.")
            break
        frames.append(df_part)

    if not frames:
        raise RuntimeError("No ETF bulk data downloaded; check FMP API key or plan.")

    raw_df = pd.concat(frames, ignore_index=True)
    norm_df = normalize_etf_holdings(raw_df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    norm_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"[ETF_BULK] Wrote {len(norm_df)} holdings rows to {OUTPUT_PATH}")


def main() -> None:
    build_etf_holdings_cache()


if __name__ == "__main__":
    main()
