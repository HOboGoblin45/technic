"""
Build a cached map of market caps for the ticker universe using Polygon.
Saves to data_cache/market_caps.json for reuse during feature building.

Usage:
    python scripts/build_market_caps.py --max-symbols 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from technic_v4.universe_loader import load_universe
from technic_v4 import data_engine

CACHE_PATH = Path("data_cache/market_caps.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache market caps for ticker universe.")
    p.add_argument("--max-symbols", type=int, default=0, help="Limit symbols (0 = all).")
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh existing entries (overwrite cached values).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    universe = load_universe()
    symbols = [u.symbol for u in universe]
    if args.max_symbols and args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]

    cache: Dict[str, float] = {}
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    added = 0
    for sym in symbols:
        sym = sym.upper()
        if sym in cache and not args.refresh:
            continue
        try:
            details = data_engine.get_ticker_details(sym)
        except Exception:
            details = {}
        mc = details.get("market_cap") if isinstance(details, dict) else None
        if mc is not None:
            cache[sym] = mc
            added += 1

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    print(
        f"Cached {len(cache)} market caps (added/updated {added}, symbols={len(symbols)}, refresh={args.refresh}) "
        f"to {CACHE_PATH}"
    )


if __name__ == "__main__":
    main()
