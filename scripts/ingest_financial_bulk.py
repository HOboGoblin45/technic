"""
Download core financial bulk files (income, balance sheet, cash flow + growth)
into data_cache/.

Usage:
  python scripts/ingest_financial_bulk.py --year 2025 --period Q1

Files written:
  data_cache/income_statement_bulk.csv
  data_cache/income_statement_growth_bulk.csv
  data_cache/balance_sheet_statement_bulk.csv
  data_cache/balance_sheet_statement_growth_bulk.csv
  data_cache/cash_flow_statement_bulk.csv
  data_cache/cash_flow_statement_growth_bulk.csv
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from io import StringIO

STABLE_BASE = "https://financialmodelingprep.com/stable"
DATA_DIR = Path("data_cache")
DATA_DIR.mkdir(parents=True, exist_ok=True)

RESOURCES = {
    "income_statement_bulk.csv": "income-statement-bulk",
    "income_statement_growth_bulk.csv": "income-statement-growth-bulk",
    "balance_sheet_statement_bulk.csv": "balance-sheet-statement-bulk",
    "balance_sheet_statement_growth_bulk.csv": "balance-sheet-statement-growth-bulk",
    "cash_flow_statement_bulk.csv": "cash-flow-statement-bulk",
    "cash_flow_statement_growth_bulk.csv": "cash-flow-statement-growth-bulk",
}


def fetch_csv(resource: str, year: int, period: str, api_key: str) -> pd.DataFrame:
    url = f"{STABLE_BASE}/{resource}"
    params = {"year": year, "period": period, "apikey": api_key}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def main():
    parser = argparse.ArgumentParser(description="Download FMP financial bulk CSVs into data_cache.")
    parser.add_argument("--year", type=int, default=date.today().year, help="Year to fetch (e.g., 2025)")
    parser.add_argument("--period", type=str, default="FY", help="Period (FY, Q1, Q2, Q3, Q4)")
    args = parser.parse_args()

    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise RuntimeError("FMP_API_KEY is not set")

    for filename, resource in RESOURCES.items():
        try:
            df = fetch_csv(resource, args.year, args.period, api_key)
            out_path = DATA_DIR / filename
            df.to_csv(out_path, index=False)
            print(f"[financial_bulk] Wrote {out_path} (rows={len(df)})")
        except Exception as exc:
            print(f"[financial_bulk] WARN failed {resource}: {exc}")


if __name__ == "__main__":
    main()
