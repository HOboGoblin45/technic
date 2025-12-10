"""
Lightweight scan quality monitor.

Checks the latest scan output for basic health and institutional constraints,
and emits a summary plus alerts. Exits with code 1 on hard failures.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


MAIN_PATH = Path("technic_v4/scanner_output/technic_scan_results.csv")
RUNNERS_PATH = Path("technic_v4/scanner_output/technic_runners.csv")
HISTORY_DIR = Path("technic_v4/scanner_output/history")

# Hard filters
MIN_PRICE = 5.0
MIN_DOLLAR_VOL = 5_000_000
MAX_ATR_PCT = 0.15
MIN_MARKET_CAP = 300_000_000


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    return df


def _hard_checks(df: pd.DataFrame) -> List[str]:
    alerts: List[str] = []
    if "Close" in df.columns:
        bad = df["Close"] < MIN_PRICE
        if bad.any():
            alerts.append(f"{bad.sum()} rows with Close < ${MIN_PRICE:.0f}")
    if {"Close", "Volume"}.issubset(df.columns):
        dv = df["Close"] * df["Volume"]
        bad = dv < MIN_DOLLAR_VOL
        if bad.any():
            alerts.append(f"{bad.sum()} rows with DollarVolume < ${MIN_DOLLAR_VOL:,.0f}")
    if "ATR14_pct" in df.columns:
        bad = pd.to_numeric(df["ATR14_pct"], errors="coerce") > MAX_ATR_PCT + 1e-9
        if bad.any():
            alerts.append(f"{bad.sum()} rows with ATR14_pct > {MAX_ATR_PCT:.2f}")
    if "market_cap" in df.columns:
        bad = pd.to_numeric(df["market_cap"], errors="coerce") < MIN_MARKET_CAP
        if bad.any():
            alerts.append(f"{bad.sum()} rows with market_cap < ${MIN_MARKET_CAP:,.0f}")
    if "IsUltraRisky" in df.columns:
        risky = df["IsUltraRisky"].fillna(False)
        if risky.any():
            alerts.append(f"{risky.sum()} ultra-risky rows present in main list")
    return alerts


def _coverage(df: pd.DataFrame) -> Dict[str, float]:
    cov: Dict[str, float] = {}
    for col in [
        "InstitutionalCoreScore",
        "AlphaScorePct",
        "TechRating",
        "market_cap",
        "next_earnings_date",
        "last_earnings_date",
        "dividend_ex_date",
    ]:
        cov[col] = df[col].notna().mean() if col in df.columns else 0.0
    return cov


def _top_metrics(df: pd.DataFrame) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    top_df = df.copy()
    if "InstitutionalCoreScore" in top_df.columns:
        top_df = top_df.sort_values("InstitutionalCoreScore", ascending=False)
    elif "TechRating" in top_df.columns:
        top_df = top_df.sort_values("TechRating", ascending=False)
    top10 = top_df.head(10)
    if not top10.empty:
        for col in ["InstitutionalCoreScore", "TechRating", "AlphaScorePct"]:
            if col in top10.columns:
                metrics[f"top10_median_{col}"] = float(pd.to_numeric(top10[col], errors="coerce").median())
    return metrics


def _sector_concentration(df: pd.DataFrame) -> Tuple[str, float]:
    if "Sector" not in df.columns or df.empty:
        return ("", 0.0)
    counts = df["Sector"].fillna("UNK").value_counts(normalize=True)
    top_sector = counts.index[0]
    top_share = float(counts.iloc[0])
    return top_sector, top_share


def _recent_history_stats() -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if not HISTORY_DIR.exists():
        return stats
    files = sorted(HISTORY_DIR.glob("scan_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)[:7]
    med_ics: List[float] = []
    med_alpha: List[float] = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty:
            continue
        if "InstitutionalCoreScore" in df.columns:
            med_ics.append(pd.to_numeric(df["InstitutionalCoreScore"], errors="coerce").median())
        if "AlphaScorePct" in df.columns:
            med_alpha.append(pd.to_numeric(df["AlphaScorePct"], errors="coerce").median())
    if med_ics:
        stats["history_median_ICS"] = float(pd.Series(med_ics).median())
    if med_alpha:
        stats["history_median_AlphaScorePct"] = float(pd.Series(med_alpha).median())
    return stats


def main() -> int:
    errors: List[str] = []
    warnings: List[str] = []

    try:
        main_df = _load_csv(MAIN_PATH)
    except Exception as exc:
        print(f"ERROR: failed to load main results: {exc}")
        return 1

    try:
        runners_df = _load_csv(RUNNERS_PATH)
    except Exception as exc:
        warnings.append(f"Runners load failed: {exc}")
        runners_df = pd.DataFrame()

    # Hard checks on main list
    hard_alerts = _hard_checks(main_df)
    errors.extend(hard_alerts)

    # Coverage
    cov = _coverage(main_df)
    top_missing = [k for k, v in cov.items() if v < 0.5]
    if top_missing:
        warnings.append(f"Low coverage on: {', '.join(top_missing)}")

    # Top metrics
    metrics = _top_metrics(main_df)

    # Sector concentration
    sector, share = _sector_concentration(main_df)
    if share > 0.4:
        warnings.append(f"Sector concentration high: {sector} at {share:.0%}")

    # History drift
    hist = _recent_history_stats()
    if hist and "history_median_ICS" in hist and "top10_median_InstitutionalCoreScore" in metrics:
        curr = metrics["top10_median_InstitutionalCoreScore"]
        prev = hist["history_median_ICS"]
        if prev > 0 and curr < 0.8 * prev:
            warnings.append(f"Top10 ICS median dropped vs history ({curr:.1f} vs {prev:.1f})")

    # Runners sanity
    if not runners_df.empty and "IsUltraRisky" in runners_df.columns:
        frac_ultra = runners_df["IsUltraRisky"].fillna(False).mean()
        if frac_ultra < 0.5:
            warnings.append("Runners list is not mostly ultra-risky.")

    # Summary output
    print(f"[monitor] Main rows: {len(main_df)}, columns: {len(main_df.columns)}")
    if runners_df is not None:
        print(f"[monitor] Runners rows: {len(runners_df)}")
    print(f"[monitor] Coverage: " + ", ".join(f"{k}={v*100:.0f}%" for k, v in cov.items()))
    if metrics:
        print("[monitor] Top metrics: " + ", ".join(f"{k}={v:.1f}" for k, v in metrics.items()))
    if sector:
        print(f"[monitor] Sector concentration: {sector} {share:.0%}")
    if hist:
        print("[monitor] History: " + ", ".join(f"{k}={v:.1f}" for k, v in hist.items()))

    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
