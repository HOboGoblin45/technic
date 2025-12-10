"""
Ingest analyst ratings and price targets into data_cache/ratings_targets.csv.

Data sources (FMP Ultimate):
  - /api/v3/rating/{symbol}                    (latest rating snapshot)
  - /api/v3/price-target-summary/{symbol}      (avg/high/low target)

We pull the latest snapshot per symbol and persist a compact CSV with:
  symbol, rating_score, rating_recommendation, rating_date,
  pt_avg, pt_high, pt_low, pt_median, pt_count, pt_currency, pt_updated

Supports chunking, append/resume, and optional sleep between symbols.

Usage examples:
  python scripts/ingest_ratings_targets.py
  python scripts/ingest_ratings_targets.py --start 0 --limit 500 --append
  python scripts/ingest_ratings_targets.py --symbols AAPL MSFT --sleep 0.1
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Set

import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = DATA_DIR / "ratings_targets.csv"

FMP_API_KEY = os.getenv("FMP_API_KEY", "")
FMP_BASE = "https://financialmodelingprep.com/api/v3"


def _require_key() -> str:
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY is not set")
    return FMP_API_KEY


def _get_json(url: str, params: Dict) -> dict:
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def fetch_rating(symbol: str) -> Dict:
    url = f"{FMP_BASE}/rating/{symbol}"
    data = _get_json(url, {"apikey": _require_key()})
    if isinstance(data, list) and data:
        return data[0] or {}
    return {}


def fetch_price_targets(symbol: str) -> Dict:
    url = f"{FMP_BASE}/price-target-summary/{symbol}"
    data = _get_json(url, {"apikey": _require_key()})
    if isinstance(data, dict):
        return data
    return {}


def run(symbols: Iterable[str], append: bool = False, sleep_sec: float = 0.0, start: int = 0, limit: int = 0) -> None:
    fieldnames = [
        "symbol",
        "rating_score",
        "rating_recommendation",
        "rating_date",
        "pt_avg",
        "pt_high",
        "pt_low",
        "pt_median",
        "pt_count",
        "pt_currency",
        "pt_updated",
    ]
    symbols = list(symbols)
    if start > 0 or limit > 0:
        end = start + limit if limit > 0 else None
        symbols = symbols[start:end]

    existing: Set[str] = set()
    mode = "a" if append and OUT_CSV.exists() else "w"
    if mode == "a":
        try:
            df_existing = pd.read_csv(OUT_CSV)
            existing = set(df_existing["symbol"].astype(str).str.upper())
        except Exception:
            existing = set()

    with OUT_CSV.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        for sym in symbols:
            sym_u = sym.upper()
            if sym_u in existing:
                continue
            try:
                rating = fetch_rating(sym_u)
            except Exception as exc:
                print(f"[ratings] WARN rating failed for {sym_u}: {exc}")
                rating = {}
            try:
                targets = fetch_price_targets(sym_u)
            except Exception as exc:
                print(f"[ratings] WARN price targets failed for {sym_u}: {exc}")
                targets = {}

            row = {
                "symbol": sym_u,
                "rating_score": rating.get("ratingScore"),
                "rating_recommendation": rating.get("ratingRecommendation"),
                "rating_date": rating.get("date"),
                "pt_avg": targets.get("priceTargetAverage"),
                "pt_high": targets.get("priceTargetHigh"),
                "pt_low": targets.get("priceTargetLow"),
                "pt_median": targets.get("priceTargetMedian"),
                "pt_count": targets.get("analystCount"),
                "pt_currency": targets.get("currency"),
                "pt_updated": targets.get("updatedAt") or targets.get("date"),
            }
            writer.writerow(row)
            print(f"[ratings] Wrote row for {sym_u}")
            if sleep_sec > 0:
                time.sleep(sleep_sec)

    print(f"[ratings] Wrote {OUT_CSV} (mode={mode}, rows={len(symbols)})")


def load_universe_symbols() -> List[str]:
    try:
        from technic_v4.universe_loader import load_universe
    except Exception as exc:
        raise RuntimeError(f"Failed to import universe_loader: {exc}")
    rows = load_universe()
    return [r.symbol for r in rows]


def main():
    parser = argparse.ArgumentParser(description="Ingest ratings and price targets (FMP).")
    parser.add_argument("--symbols", nargs="*", help="Symbols to ingest (default: universe).")
    parser.add_argument("--start", type=int, default=0, help="Offset into symbols list (for chunking).")
    parser.add_argument("--limit", type=int, default=0, help="Max symbols to process (0 = no limit).")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV and skip already-present symbols.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between symbols.")
    args = parser.parse_args()

    symbols = args.symbols if args.symbols else load_universe_symbols()
    run(symbols, append=args.append, sleep_sec=args.sleep, start=args.start, limit=args.limit)


if __name__ == "__main__":
    main()
