"""
Simple CLI to run a scan with optional as-of-date replay.

Examples:
    python scripts/run_scan.py --max-symbols 500 --as-of-date 2022-06-15
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from technic_v4.scanner_core import ScanConfig, run_scan


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Technic scan (optionally as-of a past date).")
    p.add_argument("--max-symbols", type=int, default=500, help="Limit universe size.")
    p.add_argument("--lookback-days", type=int, default=150, help="Lookback window for indicators.")
    p.add_argument("--min-tech-rating", type=float, default=0.0, help="Minimum TechRating filter.")
    p.add_argument("--trade-style", type=str, default="Short-term swing", help="Trade style.")
    p.add_argument("--as-of-date", type=str, default=None, help="Replay as of this date (YYYY-MM-DD).")
    p.add_argument("--out", type=str, default="", help="Optional path to save CSV; defaults to standard output file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ScanConfig(
        max_symbols=args.max_symbols,
        lookback_days=args.lookback_days,
        min_tech_rating=args.min_tech_rating,
        trade_style=args.trade_style,
        as_of_date=args.as_of_date,
    )
    df, status = run_scan(cfg)
    print("Status:", status)
    if df is None or df.empty:
        print("No results.")
        return

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} rows to {out_path}")
    else:
        # default behavior: reuse scanner output path
        out_path = Path("technic_v4/engine/scanner_output/technic_scan_results.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
