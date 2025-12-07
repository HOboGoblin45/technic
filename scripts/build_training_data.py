import os
from datetime import date, timedelta
from typing import List

import pandas as pd

from technic_v4 import data_engine
from technic_v4.engine.feature_engine import build_features

# Simple universe for now – you can replace with your real list
UNIVERSE: List[str] = ["AA", "AAL", "AACB"]  # extend with your core universe

START_DATE = date(2022, 1, 1)
END_DATE = date(2024, 12, 31)
LOOKBACK_DAYS = 150
FWD_DAYS = 5


def build_training_rows() -> pd.DataFrame:
    rows = []

    for symbol in UNIVERSE:
        print(f"Processing {symbol}...")
        # Pull full history once
        hist = data_engine.get_price_history(symbol, days=1000, freq="daily")
        if hist is None or hist.empty:
            continue

        # Ensure we have a Date column for easier slicing
        df = hist.reset_index()  # index → column
        # After reset_index, if the index name was "Date", you'll now have a "Date" column.
        # If the index name is something else (e.g. "index"), rename it for consistency.
        if "Date" not in df.columns:
            # assume the first column is the date-like index
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)

        df = df.sort_values("Date")

        for idx in range(LOOKBACK_DAYS, len(df) - FWD_DAYS):
            as_of_row = df.iloc[idx]
            as_of_date = as_of_row["Date"].date()

            if not (START_DATE <= as_of_date <= END_DATE):
                continue

            window = df.iloc[idx - LOOKBACK_DAYS : idx + 1]
            fut_window = df.iloc[idx + 1 : idx + 1 + FWD_DAYS]

            if fut_window.empty:
                continue

            # Forward 5-day return
            as_of_close = as_of_row["Close"]
            fwd_close = fut_window["Close"].iloc[-1]
            fwd_ret = (fwd_close - as_of_close) / as_of_close

            # Features
            feats = build_features(window, fundamentals=None)
            feats = feats.to_dict()
            feats["symbol"] = symbol
            feats["as_of_date"] = as_of_date
            feats["fwd_ret_5d"] = fwd_ret

            rows.append(feats)

    return pd.DataFrame(rows)


def main():
    df = build_training_rows()
    out_path = "data/training_data.parquet"
    os.makedirs("data", exist_ok=True)
    df.to_parquet(out_path)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
