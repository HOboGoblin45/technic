from __future__ import annotations

"""
Research Lab CLI for experimental workflows.

Subcommands:
- backtest-strategy: run a lightweight daily scan simulation over a date range using cached data.
- compare-models: compare factor vs ML vs meta alpha on historical slices.
- plot-feature-importance: show simple feature importance if available.

This tool is intentionally sandboxed: it does not modify production models/registry unless explicitly extended.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import pandas as pd

from technic_v4.scanner_core import ScanConfig, run_scan
from technic_v4.engine.strategy_profiles import get_strategy_profile
from technic_v4.evaluation import metrics as eval_metrics
from technic_v4.evaluation import scoreboard as eval_scoreboard


def _daterange(start: datetime, end: datetime) -> List[pd.Timestamp]:
    return pd.date_range(start=start, end=end, freq="B").to_pydatetime().tolist()


def cmd_backtest_strategy(args: argparse.Namespace) -> None:
    """
    Simulate running a scan each day and log signals. Uses cached data only.
    """
    prof = get_strategy_profile(args.strategy_profile_name)
    cfg = ScanConfig.from_strategy_profile(prof)
    if args.max_symbols:
        cfg.max_symbols = args.max_symbols
    start = pd.to_datetime(args.start_date)
    end = pd.to_datetime(args.end_date)
    print(f"[Lab] Backtest strategy '{prof.name}' from {start.date()} to {end.date()} using cached data only.")
    all_signals: list[pd.DataFrame] = []
    for dt_val in _daterange(start, end):
        try:
            df, status = run_scan(cfg)
            if df is None or df.empty:
                continue
            df["as_of"] = pd.Timestamp(dt_val).date()
            all_signals.append(df)
        except Exception as exc:
            print(f"[Lab] Scan failed on {dt_val.date()}: {exc}")
            continue
    if not all_signals:
        print("[Lab] No signals collected; nothing to evaluate.")
        return
    combined = pd.concat(all_signals, ignore_index=True)
    eval_path = Path("data_cache") / "research_lab_signals.csv"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(eval_path, index=False)
    print(f"[Lab] Collected {len(combined)} signals across {len(all_signals)} days -> {eval_path}")
    # Simple quality metrics
    preds = pd.Series(combined.get("AlphaScore", []))
    actual = pd.Series(combined.get("RewardRisk", []))
    ic = eval_metrics.rank_ic(preds, actual)
    prec = eval_metrics.precision_at_n(preds, actual, n=min(10, len(preds)))
    print(f"[Lab] Rank IC={ic:.3f} Precision@10={prec:.3f}")


def cmd_compare_models(args: argparse.Namespace) -> None:
    """
    Compare factor vs ML vs meta alpha on a provided CSV of historical results.
    """
    if not args.input_csv or not Path(args.input_csv).exists():
        print("[Lab] --input-csv is required and must exist.")
        return
    df = pd.read_csv(args.input_csv)
    if df.empty:
        print("[Lab] Input CSV is empty.")
        return
    actual = pd.Series(df.get("RewardRisk", []))
    comparisons = {}
    for col in ["factor_alpha", "ml_alpha", "AlphaScore"]:
        if col in df.columns:
            preds = pd.Series(df[col])
            comparisons[col] = {
                "ic": eval_metrics.rank_ic(preds, actual),
                "precision@10": eval_metrics.precision_at_n(preds, actual, n=min(10, len(preds))),
            }
    print("[Lab] Model comparison:")
    for name, vals in comparisons.items():
        print(f"  {name}: IC={vals['ic']:.3f} P@10={vals['precision@10']:.3f}")


def cmd_plot_feature_importance(args: argparse.Namespace) -> None:
    """
    Placeholder: in a richer environment, load model artifacts and display importances.
    """
    print("[Lab] Feature importance plotting not implemented in this CLI stub.")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Technic Research Lab")
    sub = parser.add_subparsers(dest="command")

    backtest = sub.add_parser("backtest-strategy", help="Simulate scans over a date range using cached data.")
    backtest.add_argument("--strategy-profile-name", required=True, help="Name of strategy profile (e.g., balanced_swing)")
    backtest.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    backtest.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    backtest.add_argument("--max-symbols", type=int, default=None, help="Optional cap per scan")

    cmp_models = sub.add_parser("compare-models", help="Compare factor vs ML vs meta alpha on historical CSV.")
    cmp_models.add_argument("--input-csv", required=True, help="CSV with columns factor_alpha/ml_alpha/AlphaScore and RewardRisk.")

    sub.add_parser("plot-feature-importance", help="Stub for feature importance visualization.")

    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "backtest-strategy":
        cmd_backtest_strategy(args)
    elif args.command == "compare-models":
        cmd_compare_models(args)
    elif args.command == "plot-feature-importance":
        cmd_plot_feature_importance(args)
    else:
        print("No command specified. Use -h for help.")


if __name__ == "__main__":
    main(sys.argv[1:])
