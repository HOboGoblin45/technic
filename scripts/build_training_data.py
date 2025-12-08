import os
from typing import List

import argparse
import pandas as pd

import sys
from pathlib import Path

# --- Ensure project root is on sys.path ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from technic_v4 import data_engine
from technic_v4.engine.scoring import compute_scores
from technic_v4.universe_loader import load_universe

# Defaults for training data generation
DEFAULT_LOOKBACK_DAYS = 150
DEFAULT_FWD_DAYS = 5
DEFAULT_TRADE_STYLE = "Short-term swing"
DEFAULT_MAX_SYMBOLS = 300


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ML alpha training data.")
    p.add_argument(
        "--max-symbols",
        type=int,
        default=DEFAULT_MAX_SYMBOLS,
        help="Maximum number of symbols to include from the universe.",
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="Minimum history length before using a row.",
    )
    p.add_argument(
        "--fwd-days",
        type=int,
        default=DEFAULT_FWD_DAYS,
        help="Forward horizon in trading days for the label (fwd_ret_5d).",
    )
    p.add_argument(
        "--trade-style",
        type=str,
        default=DEFAULT_TRADE_STYLE,
        help="Trade style passed into compute_scores (e.g. 'Short-term swing').",
    )
    p.add_argument(
        "--out",
        type=str,
        default="data/training_data.parquet",
        help="Path to output parquet file.",
    )
    return p.parse_args()


def build_training_rows(args: argparse.Namespace) -> pd.DataFrame:
    """
    Build training rows for the alpha model using the real ticker universe.

    For each symbol:
      - Pull up to ~600 days of daily history
      - For each as-of date:
          * run compute_scores on history up to that date
          * compute a forward 5-day return (fwd_ret_5d)
    """
    rows: list[dict] = []

    # Load full universe and limit to max_symbols
    universe_rows = load_universe()
    symbols: List[str] = [u.symbol for u in universe_rows]
    if args.max_symbols and args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]

    print(f"Building training data for {len(symbols)} symbols (max_symbols={args.max_symbols})")

    for symbol in symbols:
        print(f"Processing {symbol}...")

        # Pull a decent amount of daily history (e.g., ~2+ years)
        # Ensure we have enough for lookback + forward window
        days = max(args.lookback_days + args.fwd_days + 50, 300)
        hist = data_engine.get_price_history(symbol, days=days, freq="daily")
        if hist is None or hist.empty:
            print(f"  no history for {symbol}, skipping.")
            continue

        # Sort by date index
        df_hist = hist.sort_index().copy()

        if "Close" not in df_hist.columns:
            print(f"  no Close column for {symbol}, skipping.")
            continue

        n_hist = len(df_hist)
        print(f"  history length: {n_hist}")

        if n_hist <= args.lookback_days + args.fwd_days:
            print(f"  not enough history for {symbol} (need > {args.lookback_days + args.fwd_days}), skipping.")
            continue

        # For each as-of index, take all history up to that point,
        # run compute_scores on that window, and use the last row as features.
        n_rows_before = len(rows)
        for idx in range(args.lookback_days, n_hist - args.fwd_days):
            window = df_hist.iloc[: idx + 1]

            scored = compute_scores(window, trade_style=args.trade_style, fundamentals=None)
            if scored is None or scored.empty:
                continue

            # latest scored row corresponds to the as-of date
            as_of_row = scored.iloc[-1]

            # as-of date and forward close
            as_of_close = float(window["Close"].iloc[-1])
            fwd_close = float(df_hist["Close"].iloc[idx + args.fwd_days])
            fwd_ret = (fwd_close - as_of_close) / as_of_close

            # ensure there's a Date field
            if "Date" in as_of_row.index:
                as_of_date = as_of_row["Date"]
            else:
                as_of_date = window.index[-1]

            feats = as_of_row.to_dict()
            feats["symbol"] = symbol
            feats["as_of_date"] = as_of_date
            feats["fwd_ret_5d"] = fwd_ret

            rows.append(feats)

        n_rows_after = len(rows)
        print(f"  added {n_rows_after - n_rows_before} training rows for {symbol}")

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    df = build_training_rows(args)
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
