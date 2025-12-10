import os
from typing import List

import argparse
import pandas as pd

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

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
DEFAULT_START_DATE = "2010-01-01"


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
        help="Forward horizon in trading days for the primary label (fwd_ret_5d).",
    )
    p.add_argument(
        "--trade-style",
        type=str,
        default=DEFAULT_TRADE_STYLE,
        help="Trade style passed into compute_scores (e.g. 'Short-term swing').",
    )
    p.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE,
        help="Earliest as-of date to include (YYYY-MM-DD).",
    )
    p.add_argument(
        "--end-date",
        type=str,
        default=pd.Timestamp.utcnow().normalize().date().isoformat(),
        help="Latest as-of date to include (YYYY-MM-DD).",
    )
    p.add_argument(
        "--history-days",
        type=int,
        default=0,
        help=(
            "Number of calendar days to request from the data source. "
            "If 0, computed from start/end dates automatically."
        ),
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Take every Nth as-of day to limit dataset size (1 = daily).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="data/training_data.parquet",
        help="Path to output parquet file.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(4, multiprocessing.cpu_count() // 2),
        help="Number of worker threads to parallelize symbols (default: half of CPUs, min 4).",
    )
    return p.parse_args()


def build_training_rows(args: argparse.Namespace) -> pd.DataFrame:
    """
    Build training rows for the alpha model using the real ticker universe.

    For each symbol:
      - Pull daily history
      - For each as-of date:
          * run compute_scores on history up to that date
          * compute forward 5-day and 10-day returns (fwd_ret_5d, fwd_ret_10d)
    """
    rows: list[dict] = []

    # Load full universe and limit to max_symbols
    universe_rows = load_universe()
    symbols: List[str] = [u.symbol for u in universe_rows]
    if args.max_symbols and args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]

    start_dt = pd.Timestamp(args.start_date).normalize()
    end_dt = pd.Timestamp(args.end_date).normalize()
    if end_dt < start_dt:
        raise ValueError("end-date must be on or after start-date")

    print(
        f"Building training data for {len(symbols)} symbols "
        f"(max_symbols={args.max_symbols}) from {start_dt.date()} to {end_dt.date()}"
    )

    # We'll always compute both 5d and 10d; use the larger horizon for loop bounds
    fwd_5 = args.fwd_days
    fwd_10 = 10
    max_fwd_days = max(fwd_5, fwd_10)

    def _process_symbol(symbol: str) -> list[dict]:
        out_rows: list[dict] = []
        # Pull enough daily history for lookback + max forward window
        span_days = (end_dt - start_dt).days + args.lookback_days + max_fwd_days + 10
        requested_days = args.history_days if args.history_days > 0 else span_days
        days = max(requested_days, args.lookback_days + max_fwd_days + 50)
        hist = data_engine.get_price_history(symbol, days=days, freq="daily")
        if hist is None or hist.empty:
            return out_rows

        df_hist = hist.sort_index().copy()
        if not isinstance(df_hist.index, pd.DatetimeIndex):
            df_hist.index = pd.to_datetime(df_hist.index)

        if "Close" not in df_hist.columns:
            return out_rows

        n_hist = len(df_hist)
        if n_hist <= args.lookback_days + max_fwd_days:
            return out_rows

        step = max(1, args.stride)
        for idx in range(args.lookback_days, n_hist - max_fwd_days, step):
            window = df_hist.iloc[: idx + 1]
            as_of_date = window.index[-1]

            if as_of_date < start_dt or as_of_date > end_dt:
                continue

            scored = compute_scores(window, trade_style=args.trade_style, fundamentals=None)
            if scored is None or scored.empty:
                continue

            as_of_row = scored.iloc[-1]
            as_of_close = float(window["Close"].iloc[-1])

            idx_5 = idx + fwd_5
            idx_10 = idx + fwd_10
            if idx_5 >= n_hist or idx_10 >= n_hist:
                continue

            fwd_close_5 = float(df_hist["Close"].iloc[idx_5])
            fwd_close_10 = float(df_hist["Close"].iloc[idx_10])

            fwd_ret_5d = (fwd_close_5 - as_of_close) / as_of_close
            fwd_ret_10d = (fwd_close_10 - as_of_close) / as_of_close

            feats = as_of_row.to_dict()
            feats["symbol"] = symbol
            feats["as_of_date"] = as_of_date
            feats["fwd_ret_5d"] = fwd_ret_5d
            feats["fwd_ret_10d"] = fwd_ret_10d
            out_rows.append(feats)
        return out_rows

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(_process_symbol, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                sym_rows = fut.result()
            except Exception as exc:
                print(f"{sym}: error {exc}")
                continue
            if sym_rows:
                rows.extend(sym_rows)
                print(f"{sym}: added {len(sym_rows)} rows")

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
