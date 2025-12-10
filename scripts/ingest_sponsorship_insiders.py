"""
Download sponsorship/insider datasets into data_cache:
  - ETF holder bulk (part=1) -> etf_holder_bulk.csv
  - Institutional ownership latest (paged) -> institutional_ownership_latest.csv
  - Insider trading latest (paged) -> insider_trading_latest.csv

Usage:
  python scripts/ingest_sponsorship_insiders.py --inst-pages 3 --insider-pages 3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd
import requests
from io import StringIO

DATA_DIR = Path("data_cache")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    text = r.text
    # Some FMP endpoints return JSON; try CSV first, then JSON -> DataFrame
    try:
        return pd.read_csv(StringIO(text))
    except Exception:
        try:
            data = r.json()
            return pd.json_normalize(data)
        except Exception:
            raise


def fetch_paged(base_url: str, pages: int, limit: int, api_key: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in range(pages):
        url = f"{base_url}?page={p}&limit={limit}&apikey={api_key}"
        try:
            frames.append(fetch_csv(url))
        except Exception as exc:
            print(f"[sponsorship] WARN failed page {p}: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Download sponsorship and insider datasets.")
    parser.add_argument("--inst-pages", type=int, default=3, help="Pages to fetch for institutional ownership latest")
    parser.add_argument("--insider-pages", type=int, default=3, help="Pages to fetch for insider trading latest")
    parser.add_argument("--limit", type=int, default=100, help="Rows per page for paged endpoints")
    args = parser.parse_args()

    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise RuntimeError("FMP_API_KEY is not set")

    # ETF holder bulk (part=1)
    try:
        etf_url = f"https://financialmodelingprep.com/stable/etf-holder-bulk?part=1&apikey={api_key}"
        df = fetch_csv(etf_url)
        out = DATA_DIR / "etf_holder_bulk.csv"
        df.to_csv(out, index=False)
        print(f"[sponsorship] Wrote {out} (rows={len(df)})")
    except Exception as exc:
        print(f"[sponsorship] WARN etf-holder-bulk failed: {exc}")

    # Institutional ownership latest (paged)
    inst_base = "https://financialmodelingprep.com/stable/institutional-ownership/latest"
    df_inst = fetch_paged(inst_base, args.inst_pages, args.limit, api_key)
    if not df_inst.empty:
        out = DATA_DIR / "institutional_ownership_latest.csv"
        df_inst.to_csv(out, index=False)
        print(f"[sponsorship] Wrote {out} (rows={len(df_inst)})")

    # Insider trading latest (paged)
    insider_base = "https://financialmodelingprep.com/stable/insider-trading/latest"
    df_insider = fetch_paged(insider_base, args.insider_pages, args.limit, api_key)
    if not df_insider.empty:
        out = DATA_DIR / "insider_trading_latest.csv"
        df_insider.to_csv(out, index=False)
        print(f"[sponsorship] Wrote {out} (rows={len(df_insider)})")


if __name__ == "__main__":
    main()
