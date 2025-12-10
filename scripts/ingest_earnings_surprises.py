"""
Download earnings surprises in bulk and write to data_cache/earnings_surprises_bulk.csv.

Uses FMP stable bulk endpoint:
  https://financialmodelingprep.com/stable/earnings-surprises-bulk?year=YYYY&apikey=KEY

Example:
  python scripts/ingest_earnings_surprises.py --year 2025
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = DATA_DIR / "earnings_surprises_bulk.csv"

STABLE_URL = "https://financialmodelingprep.com/stable/earnings-surprises-bulk"


def download_earnings_surprises(year: int) -> pd.DataFrame:
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise RuntimeError("FMP_API_KEY is not set")

    url = f"{STABLE_URL}?year={year}&apikey={api_key}"
    df = pd.read_csv(url)
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def main():
    parser = argparse.ArgumentParser(description="Ingest earnings surprises (bulk) into data_cache.")
    parser.add_argument(
        "--year",
        type=int,
        default=date.today().year,
        help="Year to fetch (default: current year).",
    )
    args = parser.parse_args()

    df = download_earnings_surprises(args.year)
    df.to_csv(OUT_CSV, index=False)
    print(f"[earnings_surprises] Wrote {OUT_CSV} (rows={len(df)})")


if __name__ == "__main__":
    main()
