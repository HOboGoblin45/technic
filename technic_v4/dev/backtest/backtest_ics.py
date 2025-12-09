from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from typing import List

import pandas as pd

from technic_v4 import data_engine
from technic_v4.scanner_core import OUTPUT_DIR


HISTORY_DIR = OUTPUT_DIR / "history"


def _parse_scan_date(scan_date_str: str) -> datetime:
    """
    Parse the scan_date field or filename-derived date into a datetime.
    """
    # scan_date might be '2025-12-08' or full ISO
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(scan_date_str, fmt)
        except Exception:
            continue
    # Fallback: just use date part if present
    try:
        return datetime.fromisoformat(scan_date_str)
    except Exception:
        return datetime.utcnow()


def _compute_forward_returns(symbol: str, as_of: datetime, days_list: List[int]) -> dict:
    """
    For a given symbol and reference date, compute realized forward returns for several horizons.
    Returns a dict like {'fwd_ret_5d': ..., 'fwd_ret_10d': ..., ...}.
    """
    # Pull ~60 trading days of daily history around as_of
    hist = data_engine.get_price_history(symbol=symbol, days=90, freq="daily")
    if hist is None or hist.empty:
        return {f"fwd_ret_{d}d": float("nan") for d in days_list}

    # Ensure index is datetime-like
    df = hist.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df.index = pd.to_datetime(df["Date"])
        else:
            df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    # Find the row at or just before as_of date
    as_of_date = as_of.date()
    df_asof = df[df.index.date <= as_of_date]
    if df_asof.empty:
        return {f"fwd_ret_{d}d": float("nan") for d in days_list}

    anchor_ts = df_asof.index.max()
    anchor_close = float(df.loc[anchor_ts, "Close"])

    out = {}
    for d in days_list:
        target_date = anchor_ts + timedelta(days=d)
        # choose the first available bar on or after target_date
        df_fwd = df[df.index >= target_date]
        if df_fwd.empty:
            out[f"fwd_ret_{d}d"] = float("nan")
        else:
            fwd_close = float(df_fwd.iloc[0]["Close"])
            out[f"fwd_ret_{d}d"] = (fwd_close / anchor_close) - 1.0
    return out


def load_scan_history() -> pd.DataFrame:
    """
    Load all scan_*.csv snapshots from HISTORY_DIR and combine into a single DataFrame.
    Adds 'scan_date' as a column.
    """
    if not HISTORY_DIR.exists():
        raise FileNotFoundError(f"History directory does not exist: {HISTORY_DIR}")

    rows = []
    for path in sorted(HISTORY_DIR.glob("scan_*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        # derive scan_date from filename if no explicit column exists
        name = path.stem  # e.g. 'scan_2025-12-08'
        parts = name.split("_", 1)
        file_date = parts[1] if len(parts) == 2 else ""

        if "scan_date" in df.columns:
            scan_date_raw = df["scan_date"].iloc[0]
            scan_dt = _parse_scan_date(str(scan_date_raw))
        else:
            scan_dt = _parse_scan_date(file_date)

        df["scan_date"] = scan_dt.date().isoformat()
        rows.append(df)

    if not rows:
        raise ValueError(f"No historical scan CSVs found in {HISTORY_DIR}")

    all_scans = pd.concat(rows, ignore_index=True)
    return all_scans


def backtest_ics_buckets(
    df: pd.DataFrame,
    horizons: List[int] = [5, 10, 21],
    n_buckets: int = 5,
) -> pd.DataFrame:
    """
    For each historical scan row, compute forward returns for given horizons,
    then group by InstitutionalCoreScore buckets and average the returns.
    """
    if "InstitutionalCoreScore" not in df.columns:
        raise ValueError("InstitutionalCoreScore column missing from scan history.")

    # Compute fwd returns per row
    fwd_rows = []
    for idx, row in df.iterrows():
        symbol = str(row.get("Symbol") or "")
        if not symbol:
            continue

        scan_date_str = row.get("scan_date") or ""
        scan_dt = _parse_scan_date(str(scan_date_str))
        rets = _compute_forward_returns(symbol, scan_dt, horizons)
        rets["Symbol"] = symbol
        rets["scan_date"] = scan_dt.date().isoformat()
        rets["InstitutionalCoreScore"] = row.get("InstitutionalCoreScore", float("nan"))
        fwd_rows.append(rets)

    if not fwd_rows:
        raise ValueError("No forward-return rows computed; check history or data source.")

    fwd_df = pd.DataFrame(fwd_rows)

    # Drop rows with missing ICS or all NaN returns
    mask_valid = fwd_df["InstitutionalCoreScore"].notna()
    for d in horizons:
        mask_valid &= fwd_df[f"fwd_ret_{d}d"].notna()

    fwd_df = fwd_df[mask_valid]
    if fwd_df.empty:
        raise ValueError("No valid rows with both ICS and forward returns.")

    # Bucket by ICS into quantiles
    fwd_df["ICS_bucket"] = pd.qcut(
        fwd_df["InstitutionalCoreScore"],
        q=n_buckets,
        labels=False,
        duplicates="drop",
    )

    group_cols = ["ICS_bucket"]
    agg = {f"fwd_ret_{d}d": "mean" for d in horizons}
    result = fwd_df.groupby(group_cols).agg(agg).reset_index()

    # Sort buckets from lowest ICS to highest
    result = result.sort_values("ICS_bucket")
    return result


def main():
    print(f"Loading scan history from: {HISTORY_DIR}")
    history_df = load_scan_history()
    print(f"Loaded {len(history_df)} rows from history.")

    horizons = [5, 10, 21]
    result = backtest_ics_buckets(history_df, horizons=horizons, n_buckets=5)

    print("\nAverage realized forward returns by ICS bucket:")
    print("(Bucket 0 = lowest ICS, highest = best ideas)")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
