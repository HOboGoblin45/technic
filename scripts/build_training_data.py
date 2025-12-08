import os
from typing import List

import pandas as pd

from technic_v4 import data_engine
from technic_v4.engine.scoring import compute_scores

# Small training universe to start; expand once this is working
UNIVERSE: List[str] = ["AA", "AAL", "AACB"]

LOOKBACK_DAYS = 150   # minimum history before we trust scores
FWD_DAYS = 5
TRADE_STYLE = "Short-term swing"


def build_training_rows() -> pd.DataFrame:
    rows = []

    for symbol in UNIVERSE:
        print(f"Processing {symbol}...")

        # Pull a decent amount of daily history (e.g., ~2+ years)
        hist = data_engine.get_price_history(symbol, days=600, freq="daily")
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

        # For each as-of index, take all history up to that point,
        # run compute_scores on that window, and use the last row as features.
        n_rows_before = len(rows)
        for idx in range(LOOKBACK_DAYS, n_hist - FWD_DAYS):
            window = df_hist.iloc[: idx + 1]

            scored = compute_scores(window, trade_style=TRADE_STYLE, fundamentals=None)
            if scored is None or scored.empty:
                continue

            # latest scored row corresponds to the as-of date
            as_of_row = scored.iloc[-1]

            # as-of date and forward close
            as_of_close = float(window["Close"].iloc[-1])
            fwd_close = float(df_hist["Close"].iloc[idx + FWD_DAYS])
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
    df = build_training_rows()
    out_path = "data/training_data.parquet"
    os.makedirs("data", exist_ok=True)
    df.to_parquet(out_path)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
