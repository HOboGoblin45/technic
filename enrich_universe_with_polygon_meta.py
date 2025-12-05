import os
import time
import csv
import requests
from typing import Dict, Any


API_KEY = os.getenv("POLYGON_API_KEY") or "YOUR_POLYGON_API_KEY_HERE"
BASE_URL = "https://api.polygon.io"
INPUT_UNIVERSE = "technic_v4/ticker_universe.csv"
OUTPUT_UNIVERSE = "ticker_universe_enriched.csv"

SLEEP_SECONDS = 0.25  # throttle a bit to be kind to the API


def fetch_ticker_details(symbol: str) -> Dict[str, Any]:
    """Fetch ticker details from Polygon v3/reference/tickers/{symbol}."""
    url = f"{BASE_URL}/v3/reference/tickers/{symbol}"
    params = {"apiKey": API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=10)
    except Exception as e:
        print(f"[META ERROR] {symbol}: request failed: {e}")
        return {}

    if resp.status_code != 200:
        print(f"[META ERROR] {symbol}: HTTP {resp.status_code} -> {resp.text[:200]}")
        return {}

    data = resp.json() or {}
    results = data.get("results") or {}

    return {
        "Name": results.get("name") or "",
        "MarketCap": results.get("market_cap"),
        "SicCode": results.get("sic_code"),
        "SicDescription": results.get("sic_description") or "",
    }


def main() -> None:
    if not API_KEY or API_KEY == "YOUR_POLYGON_API_KEY_HERE":
        raise SystemExit(
            "Please set your Polygon API key in the POLYGON_API_KEY environment "
            "variable or in this file (API_KEY)."
        )

    if not os.path.exists(INPUT_UNIVERSE):
        raise SystemExit(f"Input universe file not found: {INPUT_UNIVERSE}")

    with open(INPUT_UNIVERSE, "r", newline="") as f:
        reader = csv.DictReader(f)
        symbols = [row["symbol"].strip().upper() for row in reader if row.get("symbol")]

    symbols = sorted(set(s for s in symbols if s))

    print(f"[UNIVERSE] {len(symbols)} symbols found in {INPUT_UNIVERSE}")

    rows_out = []
    for i, sym in enumerate(symbols, start=1):
        print(f"[{i}/{len(symbols)}] Fetching meta for {sym}...")
        meta = fetch_ticker_details(sym)

        # Fallbacks if Polygon returns nothing
        name = meta.get("Name", "")
        sic_desc = meta.get("SicDescription", "")
        sic_code = meta.get("SicCode")
        mcap = meta.get("MarketCap")

        # For now, use SIC description as "Industry" and leave Sector empty/Unknown.
        sector = ""  # you can add your own mapping from SIC -> sector later
        industry = sic_desc

        rows_out.append(
            {
                "symbol": sym,
                "Name": name,
                "Sector": sector,
                "Industry": industry,
                "MarketCap": mcap,
                "SicCode": sic_code,
                "SicDescription": sic_desc,
            }
        )

        # gentle throttle
        time.sleep(SLEEP_SECONDS)

    fieldnames = [
        "symbol",
        "Name",
        "Sector",
        "Industry",
        "MarketCap",
        "SicCode",
        "SicDescription",
    ]

    with open(OUTPUT_UNIVERSE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[DONE] Wrote enriched universe to {OUTPUT_UNIVERSE}")


if __name__ == "__main__":
    main()
