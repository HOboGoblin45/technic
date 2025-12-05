"""
Ingest fundamentals into data_cache/fundamentals/ from a provider.

Usage:
    python scripts/ingest_fundamentals.py symbols.txt [provider]

Providers:
    - fmp (FinancialModelingPrep)   env: FMP_API_KEY
    - polygon                       env: POLYGON_API_KEY
    - yf (fallback yfinance)

Fields written per symbol:
    pe, peg, piotroski, altman_z, eps_growth
"""

from __future__ import annotations

import sys
import json
import os
from pathlib import Path
from typing import Any, Iterable

import requests

try:
    import yfinance as yf
except ImportError:
    yf = None

ROOT = Path(__file__).resolve().parents[1]
FUND_DIR = ROOT / "data_cache" / "fundamentals"
FUND_DIR.mkdir(parents=True, exist_ok=True)


def _pick_first(source: dict[str, Any], keys: Iterable[str]) -> Any:
    """Return the first non-null value for the given keys."""
    for k in keys:
        if k in source and source[k] not in (None, "", "None"):
            return source[k]
    return None


def fetch_fmp(symbol: str, api_key: str) -> dict:
    """
    Pull a compact fundamental snapshot from FinancialModelingPrep.
    We stitch together a few endpoints to cover the required fields.
    """
    base = "https://financialmodelingprep.com/api/v3"
    out: dict[str, Any] = {}

    # --- Key metrics (TTM) for PE / PEG / EPS growth --------------------
    try:
        url = f"{base}/key-metrics-ttm/{symbol}?apikey={api_key}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            m = data[0]
            out["pe"] = _pick_first(m, ["peRatioTTM", "peRatio"])
            out["peg"] = _pick_first(m, ["pegRatioTTM", "pegRatio"])
            out["eps_growth"] = _pick_first(
                m,
                [
                    "fiveYNetIncomeGrowthPerShare",
                    "fiveYEpsGrowthPerShare",
                    "netIncomeGrowthRate",
                ],
            )
    except Exception:
        pass

    # --- Ratios TTM (Altman Z) -----------------------------------------
    try:
        url = f"{base}/ratios-ttm/{symbol}?apikey={api_key}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            ratios = data[0]
            out["altman_z"] = _pick_first(ratios, ["altmanZScore", "altmanZScoreTTM"])
            # Sometimes EPS growth lives here too
            if "epsGrowthTTM" in ratios and ratios["epsGrowthTTM"] is not None:
                out.setdefault("eps_growth", ratios["epsGrowthTTM"])
    except Exception:
        pass

    # --- Piotroski score ------------------------------------------------
    try:
        url = f"{base}/score/{symbol}?apikey={api_key}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            out["piotroski"] = data[0].get("score")
    except Exception:
        pass

    # --- Fallback EPS growth from financial-growth ---------------------
    try:
        url = f"{base}/financial-growth/{symbol}?limit=1&apikey={api_key}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            g = data[0]
            out.setdefault(
                "eps_growth",
                _pick_first(g, ["epsGrowth", "epsgrowth", "epsGrowthTTM"]),
            )
    except Exception:
        pass

    return out


def fetch_polygon(symbol: str, api_key: str) -> dict:
    """
    Polygon (Massive) fundamentals snapshot.
    Note: Polygon does not expose Piotroski/Altman Z directly;
    we pull what is available (PE / PEG / EPS growth) and leave
    the rest as None for downstream fallbacks.
    """
    out: dict[str, Any] = {}
    try:
        url = f"https://api.polygon.io/vX/reference/financials/{symbol}?limit=1&apiKey={api_key}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data and data.get("results"):
            res = data["results"][0]
            fs = res.get("financials") or {}
            ratios = fs.get("ratios", {})
            out["pe"] = ratios.get("priceToEarnings")
            out["peg"] = ratios.get("pegRatio")
            out["eps_growth"] = ratios.get("earningsGrowth")
    except Exception:
        pass
    return out


def fetch_yf(symbol: str) -> dict:
    if yf is None:
        return {}
    tkr = yf.Ticker(symbol)
    info = tkr.info or {}
    return {
        "pe": info.get("trailingPE"),
        "peg": info.get("pegRatio"),
        "piotroski": None,
        "altman_z": None,
        "eps_growth": info.get("earningsQuarterlyGrowth"),
    }


def fetch(symbol: str, provider: str) -> dict:
    sym = symbol.upper()
    if provider == "fmp":
        key = os.getenv("FMP_API_KEY")
        if not key:
            print("FMP_API_KEY not set; falling back to yfinance")
            return fetch_yf(sym)
        return fetch_fmp(sym, key)
    if provider == "polygon":
        key = os.getenv("POLYGON_API_KEY")
        if not key:
            print("POLYGON_API_KEY not set; falling back to yfinance")
            return fetch_yf(sym)
        return fetch_polygon(sym, key)
    return fetch_yf(sym)


def ingest(symbols: list[str], provider: str) -> None:
    for sym in symbols:
        sym = sym.strip().upper()
        if not sym:
            continue
        data = fetch(sym, provider)
        for key in ("pe", "peg", "piotroski", "altman_z", "eps_growth"):
            data.setdefault(key, None)
        dest = FUND_DIR / f"{sym}.json"
        dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Wrote {dest}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_fundamentals.py symbols.txt [provider]")
        sys.exit(1)

    symbols_file = Path(sys.argv[1])
    if not symbols_file.exists():
        print(f"Symbols file not found: {symbols_file}")
        sys.exit(1)

    provider = sys.argv[2] if len(sys.argv) > 2 else "yf"
    symbols = symbols_file.read_text().splitlines()
    ingest(symbols, provider)
