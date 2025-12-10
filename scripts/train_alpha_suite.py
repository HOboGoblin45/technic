"""
Convenience runner to train multiple alpha models (base/regime/sector/rolling) in one go.

Examples:
    python scripts/train_alpha_suite.py --train-path data/training_data_v2.parquet --label fwd_ret_5d
    python scripts/train_alpha_suite.py --train-path data/training_data_v2.parquet --label fwd_ret_10d --rolling --regime-split --sector-split
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multiple alpha model variants (base/regime/sector/rolling).")
    p.add_argument("--train-path", type=str, required=True, help="Path to training data parquet.")
    p.add_argument("--label", type=str, default="fwd_ret_5d", help="Label column to train on.")
    p.add_argument("--base", action="store_true", help="Train base model (no splits).")
    p.add_argument("--regime-split", action="store_true", help="Train regime-specific models.")
    p.add_argument("--sector-split", action="store_true", help="Train sector-specific models.")
    p.add_argument("--rolling", action="store_true", help="Train rolling-window models.")
    p.add_argument("--rolling-start", type=str, default="2010-01-01", help="Start date for rolling windows.")
    p.add_argument("--rolling-train-years", type=int, default=8, help="Training window length in years.")
    p.add_argument("--rolling-val-years", type=int, default=1, help="Validation window length in years.")
    p.add_argument("--rolling-test-years", type=int, default=1, help="Test window length in years.")
    p.add_argument("--rolling-step-years", type=int, default=1, help="Step size between rolling windows in years.")
    p.add_argument(
        "--model-prefix",
        type=str,
        default="models/alpha/xgb_v2",
        help="Base path prefix for model outputs (suffixes will be added).",
    )
    p.add_argument(
        "--skip-base-if-rolling",
        action="store_true",
        help="Skip base training when rolling is enabled (to save time).",
    )
    p.add_argument(
        "--skip-base-if-splits",
        action="store_true",
        help="Skip base training when regime/sector splits are requested.",
    )
    return p.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    base_cmd = [
        sys.executable,
        "scripts/train_alpha_xgb.py",
        "--train-path",
        args.train_path,
        "--label",
        args.label,
    ]

    # Decide whether to run the base model
    run_base = args.base or (not args.regime_split and not args.sector_split and not args.rolling)
    if args.rolling and args.skip_base_if_rolling:
        run_base = False
    if (args.regime_split or args.sector_split) and args.skip_base_if_splits:
        run_base = False

    if run_base:
        run_cmd(base_cmd + ["--model-path", f"{args.model_prefix}.pkl"])

    if args.regime_split:
        run_cmd(
            base_cmd
            + [
                "--model-path",
                f"{args.model_prefix}.pkl",
                "--regime-split",
            ]
        )

    if args.sector_split:
        run_cmd(
            base_cmd
            + [
                "--model-path",
                f"{args.model_prefix}.pkl",
                "--sector-split",
            ]
        )

    if args.rolling:
        run_cmd(
            base_cmd
            + [
                "--model-path",
                f"{args.model_prefix}.pkl",
                "--rolling",
                "--rolling-start",
                args.rolling_start,
                "--rolling-window-train-years",
                str(args.rolling_train_years),
                "--rolling-window-val-years",
                str(args.rolling_val_years),
                "--rolling-window-test-years",
                str(args.rolling_test_years),
                "--rolling-step-years",
                str(args.rolling_step_years),
            ]
            + (["--regime-split"] if args.regime_split else [])
            + (["--sector-split"] if args.sector_split else [])
        )


if __name__ == "__main__":
    main()
