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
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = DATA_DIR / "ratings_targets.csv"

FMP_API_KEY = os.getenv("FMP_API_KEY", "")
FMP_BASE = "https://financialmodelingprep.com/api/v3"
FMP_BASE_V4 = "https://financialmodelingprep.com/api/v4"
STABLE_BASE = "https://financialmodelingprep.com/stable"

# Local bulk cache files (written by prior downloads)
RATINGS_BULK_CSV = DATA_DIR / "ratings_bulk.csv"
PRICE_TARGETS_BULK_CSV = DATA_DIR / "price_targets_bulk.csv"


def _require_key() -> str:
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY is not set")
    return FMP_API_KEY


def _get_json(url: str, params: Dict) -> dict:
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _read_bulk_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None


def _download_stable_csv(resource: str) -> Optional[pd.DataFrame]:
    """
    Fetch a stable bulk CSV directly to a DataFrame.
    """
    if not FMP_API_KEY:
        return None
    url = f"{STABLE_BASE}/{resource}?apikey={FMP_API_KEY}"
    try:
        df = pd.read_csv(url)
        return df
    except Exception:
        return None


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


def fetch_bulk_ratings() -> pd.DataFrame:
    """
    Prefer local stable CSV (rating-bulk); fallback to live stable download; lastly v3 JSON.
    """
    df = _read_bulk_csv(RATINGS_BULK_CSV)
    if df is None:
        df = _download_stable_csv("rating-bulk")
    if df is None:
        url = f"{FMP_BASE}/rating"
        params = {"apikey": _require_key()}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()
    if not df.empty:
        df.columns = [c.lower() for c in df.columns]
    return df


def fetch_bulk_price_targets() -> pd.DataFrame:
    """
    Prefer local stable CSV (price-target-summary-bulk); fallback to live stable download; lastly v4 JSON.
    """
    df = _read_bulk_csv(PRICE_TARGETS_BULK_CSV)
    if df is None:
        df = _download_stable_csv("price-target-summary-bulk")
    if df is None:
        url = f"{FMP_BASE_V4}/price-target-consensus"
        params = {"apikey": _require_key()}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()
    if not df.empty:
        df.columns = [c.lower() for c in df.columns]
    return df


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

    # Try bulk first (local or stable CSV)
    bulk_df = pd.DataFrame()
    pt_df = pd.DataFrame()
    try:
        bulk_df = fetch_bulk_ratings()
        if not bulk_df.empty:
            bulk_df["symbol"] = bulk_df["symbol"].astype(str).str.upper()
    except Exception as exc:
        print(f"[ratings] WARN bulk ratings failed: {exc}")
    try:
        pt_df = fetch_bulk_price_targets()
        if not pt_df.empty:
            pt_df["symbol"] = pt_df["symbol"].astype(str).str.upper()
    except Exception as exc:
        print(f"[ratings] WARN bulk price targets failed: {exc}")

    if not bulk_df.empty or not pt_df.empty:
        df_out = pd.DataFrame(columns=fieldnames)
        if not bulk_df.empty:
            df_out = bulk_df
        if not pt_df.empty:
            df_out = df_out.merge(pt_df, on="symbol", how="outer") if not df_out.empty else pt_df

        df_out = df_out.rename(
            columns={
                "rating": "rating_recommendation",
                "ratingrecomendation": "rating_recommendation",
                "ratingscore": "rating_score",
                "date": "rating_date",
                "pricetargetaverage": "pt_avg",
                "pricetargethigh": "pt_high",
                "pricetargetlow": "pt_low",
                "pricetargetmedian": "pt_median",
                "analystcount": "pt_count",
                "currency": "pt_currency",
                "updatedat": "pt_updated",
                "alltimeavgpricetarget": "pt_avg",
                "alltimecount": "pt_count",
            }
        )

        # Construct final frame
        out_rows = []
        for _, row in df_out.iterrows():
            sym = str(row.get("symbol", "")).upper()
            if not sym:
                continue
            out_rows.append(
                {
                    "symbol": sym,
                    "rating_score": row.get("rating_score"),
                    "rating_recommendation": row.get("rating_recommendation"),
                    "rating_date": row.get("rating_date"),
                    "pt_avg": row.get("pt_avg"),
                    "pt_high": row.get("pt_high"),
                    "pt_low": row.get("pt_low"),
                    "pt_median": row.get("pt_median"),
                    "pt_count": row.get("pt_count"),
                    "pt_currency": row.get("pt_currency"),
                    "pt_updated": row.get("pt_updated"),
                }
            )

        merged = pd.DataFrame(out_rows, columns=fieldnames)
        merged.to_csv(OUT_CSV, index=False)
        print(f"[ratings] Wrote bulk ratings/targets to {OUT_CSV} (rows={len(merged)})")
        return

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
