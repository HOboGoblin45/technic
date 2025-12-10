"""
Ingest fundamental quality metrics into data_cache/quality_scores.csv.

Data sources (FMP Ultimate):
  - /api/v3/financial-scores/{symbol}    (quality composite)
  - /api/v3/ratios-ttm/{symbol}          (ROE, margins, leverage)
  - /api/v3/key-metrics-ttm/{symbol}     (FCF, debt metrics)

We build a simple QualityScore (0-100) blending ROE, margins, leverage penalty,
and the FMP financialScore when available.

Columns written:
  symbol, quality_score, roe, gross_margin, op_margin, net_margin,
  debt_to_equity, interest_coverage, fcf_margin, fmp_financial_score

Usage:
  python scripts/ingest_quality_scores.py [--symbols SYMBOL1 SYMBOL2 ...]
  (default: load universe via technic_v4.universe_loader)
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = DATA_DIR / "quality_scores.csv"

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


def fetch_financial_score(symbol: str) -> Dict:
    url = f"{FMP_BASE}/financial-scores/{symbol}"
    data = _get_json(url, {"apikey": _require_key(), "limit": 1})
    if isinstance(data, list) and data:
        return data[0] or {}
    return {}


def fetch_ratios_ttm(symbol: str) -> Dict:
    url = f"{FMP_BASE}/ratios-ttm/{symbol}"
    data = _get_json(url, {"apikey": _require_key(), "limit": 1})
    if isinstance(data, list) and data:
        return data[0] or {}
    return {}


def fetch_key_metrics_ttm(symbol: str) -> Dict:
    url = f"{FMP_BASE}/key-metrics-ttm/{symbol}"
    data = _get_json(url, {"apikey": _require_key(), "limit": 1})
    if isinstance(data, list) and data:
        return data[0] or {}
    return {}


def _norm(val) -> float:
    try:
        return float(val)
    except Exception:
        return float("nan")


def compute_quality_score(
    roe: float,
    op_margin: float,
    net_margin: float,
    dte: float,
    fin_score: float,
    fcf_margin: float,
) -> float:
    import math

    parts: List[float] = []
    if not math.isnan(fin_score):
        # FMP financialScore is 0-100 already
        parts.append(fin_score)
    if not math.isnan(roe):
        parts.append(max(min(roe, 0.5), -0.5) * 100)  # cap +/-50%
    if not math.isnan(op_margin):
        parts.append(max(min(op_margin, 0.5), -0.2) * 100)
    if not math.isnan(net_margin):
        parts.append(max(min(net_margin, 0.4), -0.2) * 100)
    if not math.isnan(fcf_margin):
        parts.append(max(min(fcf_margin, 0.5), -0.2) * 100)
    if not math.isnan(dte):
        # leverage penalty: higher D/E reduces score
        pen = min(max(dte, 0.0), 5.0) / 5.0  # 0 to 1
        parts.append(50 * (1 - pen))
    if not parts:
        return float("nan")
    return max(0.0, min(100.0, sum(parts) / len(parts)))


def run(symbols: Iterable[str]) -> None:
    fieldnames = [
        "symbol",
        "quality_score",
        "roe",
        "gross_margin",
        "op_margin",
        "net_margin",
        "debt_to_equity",
        "interest_coverage",
        "fcf_margin",
        "fmp_financial_score",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sym in symbols:
            sym_u = sym.upper()
            try:
                fin = fetch_financial_score(sym_u)
            except Exception as exc:
                print(f"[quality] WARN financial_score failed for {sym_u}: {exc}")
                fin = {}
            try:
                ratios = fetch_ratios_ttm(sym_u)
            except Exception as exc:
                print(f"[quality] WARN ratios failed for {sym_u}: {exc}")
                ratios = {}
            try:
                km = fetch_key_metrics_ttm(sym_u)
            except Exception as exc:
                print(f"[quality] WARN key metrics failed for {sym_u}: {exc}")
                km = {}

            roe = _norm(ratios.get("returnOnEquityTTM"))
            gross_margin = _norm(ratios.get("grossProfitMarginTTM"))
            op_margin = _norm(ratios.get("operatingProfitMarginTTM"))
            net_margin = _norm(ratios.get("netProfitMarginTTM"))
            dte = _norm(ratios.get("debtEquityRatioTTM"))
            int_cov = _norm(ratios.get("interestCoverageTTM"))

            fcf = _norm(km.get("freeCashFlowTTM"))
            revenue = _norm(km.get("revenuePerShareTTM")) * _norm(km.get("sharesOutstanding") or float("nan"))
            if revenue and not (revenue != revenue):
                fcf_margin = fcf / revenue if revenue else float("nan")
            else:
                fcf_margin = float("nan")

            fin_score = _norm(fin.get("financialScore"))

            qscore = compute_quality_score(roe, op_margin, net_margin, dte, fin_score, fcf_margin)

            writer.writerow(
                {
                    "symbol": sym_u,
                    "quality_score": qscore,
                    "roe": roe,
                    "gross_margin": gross_margin,
                    "op_margin": op_margin,
                    "net_margin": net_margin,
                    "debt_to_equity": dte,
                    "interest_coverage": int_cov,
                    "fcf_margin": fcf_margin,
                    "fmp_financial_score": fin_score,
                }
            )
            print(f"[quality] Wrote row for {sym_u}")

    print(f"[quality] Wrote {OUT_CSV}")


def load_universe_symbols() -> List[str]:
    try:
        from technic_v4.universe_loader import load_universe
    except Exception as exc:
        raise RuntimeError(f"Failed to import universe_loader: {exc}")
    rows = load_universe()
    return [r.symbol for r in rows]


def main():
    parser = argparse.ArgumentParser(description="Ingest quality scores (FMP).")
    parser.add_argument("--symbols", nargs="*", help="Symbols to ingest (default: universe).")
    args = parser.parse_args()

    symbols = args.symbols if args.symbols else load_universe_symbols()
    run(symbols)


if __name__ == "__main__":
    main()
