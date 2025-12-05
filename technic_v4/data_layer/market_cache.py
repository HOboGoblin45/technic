# technic_v4/data_layer/market_cache.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

from technic_v4.data_layer.bulk_daily import get_daily_snapshot, CACHE_DIR


@dataclass
class MarketCacheConfig:
    lookback_days: int = 150   # how many calendar days of snapshots to build from
    extra_buffer: int = 10     # extra days to request to account for weekends/holidays


class MarketCache:
    """
    Local OHLCV history built from Polygon grouped daily snapshots.

    - Uses get_daily_snapshot(date) to ensure snapshots exist on disk.
    - Loads the last N snapshots into a single DataFrame.
    - Provides per-symbol history as a standard OHLCV DataFrame.
    """

    def __init__(self, config: Optional[MarketCacheConfig] = None) -> None:
        if config is None:
            config = MarketCacheConfig()
        self.config = config

        # This will:
        #  1) ensure daily snapshot parquet files exist for the recent window
        #  2) load and combine them into self._df
        self._df = self._build_history()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_snapshots(self) -> None:
        """
        Ensure that we have grouped daily snapshots on disk for the recent window.

        We request roughly lookback_days + extra_buffer calendar days going
        backward from today. Weekends/holidays may not have data, but that's
        fine; we'll just end up with ~lookback_days trading sessions.
        """
        end_date = datetime.utcnow().date()
        max_days = self.config.lookback_days + self.config.extra_buffer

        for offset in range(0, max_days):
            d = end_date - timedelta(days=offset)
            try:
                # This will use the cache if the file already exists
                get_daily_snapshot(d)
            except Exception as exc:
                # Non-fatal; log and continue
                print(f"[MARKETCACHE WARN] Failed snapshot for {d}: {exc}")

    def _build_history(self) -> pd.DataFrame:
        """
        Build a combined OHLCV DataFrame from the last N snapshot files.

        Returns a DataFrame with columns:
            Symbol, Date, Open, High, Low, Close, Volume, VWAP
        """
        # Make sure we have recent snapshot files cached
        self._ensure_snapshots()

        # Find all cached daily parquet files
        files: List[Path] = list(CACHE_DIR.glob("daily_*.parquet"))
        if not files:
            raise RuntimeError("No daily snapshot files found in cache directory.")

        # Sort files by date embedded in filename, keep last lookback_days files
        def _file_date(p: Path) -> datetime:
            # filenames look like daily_YYYY-MM-DD.parquet
            stem = p.stem  # e.g., "daily_2025-11-28"
            date_str = stem.replace("daily_", "")
            return datetime.strptime(date_str, "%Y-%m-%d")

        files_sorted = sorted(files, key=_file_date)
        # Take only the most recent lookback_days files (or all if fewer)
        files_selected = files_sorted[-self.config.lookback_days :]

        dfs: List[pd.DataFrame] = []
        for f in files_selected:
            try:
                df_day = pd.read_parquet(f)
                dfs.append(df_day)
            except Exception as exc:
                print(f"[MARKETCACHE WARN] Failed to read {f}: {exc}")

        if not dfs:
            raise RuntimeError("No valid daily snapshot data could be loaded.")

        df = pd.concat(dfs, ignore_index=True)

        # Ensure Date is datetime and sorted
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Symbol", "Date"])

        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_symbol_history(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """
        Return a DataFrame of OHLCV for `symbol`, up to `days` bars back.

        Columns:
            Open, High, Low, Close, Volume

        Index:
            Date (datetime)
        """
        df_sym = self._df[self._df["Symbol"] == symbol]
        if df_sym.empty:
            return None

        # Take the last `days` rows and shape into the expected format
        df_sym = df_sym.tail(days).copy()
        df_sym = df_sym.set_index("Date")

        # Keep only the standard OHLCV columns expected by indicators/scoring
        cols = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in cols if c not in df_sym.columns]
        if missing:
            print(f"[MARKETCACHE WARN] {symbol} missing columns: {missing}")
            return None

        return df_sym[cols]
