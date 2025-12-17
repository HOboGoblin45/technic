from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path("technic_v4/data_cache")
HOLDINGS_PATH = DATA_DIR / "etf_holdings.parquet"


def load_etf_holdings() -> pd.DataFrame:
    """
    Load the normalized ETF holdings table produced by build_etf_holdings_cache_fmp.

    Schema:
      - etf_symbol:   ETF ticker (e.g. SPY)
      - asset_symbol: underlying asset ticker (e.g. AAPL)
      - asset_name:   underlying asset name
      - weight_pct:   % of ETF allocated to this asset
      - shares:       number of shares held
      - market_value: dollar value of this holding
      - plus optional columns like 'isin', 'cusip', 'name'
    """
    if not HOLDINGS_PATH.exists():
        raise FileNotFoundError(
            f"ETF holdings cache not found at {HOLDINGS_PATH}. "
            "Run technic_v4/dev/data/build_etf_holdings_cache_fmp.py first."
        )
    df = pd.read_parquet(HOLDINGS_PATH)
    return df


def get_etf_holdings(etf_symbol: str, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Return all holdings rows for a given ETF symbol.

    Example: get_etf_holdings("SPY") -> all stocks SPY holds with weights & shares.
    """
    if df is None:
        df = load_etf_holdings()

    etf_symbol = etf_symbol.upper()
    sub = df.loc[df["etf_symbol"] == etf_symbol].copy()
    # Sort by descending weight
    if "weight_pct" in sub.columns:
        sub = sub.sort_values("weight_pct", ascending=False)
    return sub


def get_asset_exposure(asset_symbol: str, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Return all ETF rows that hold a given stock.

    Example: get_asset_exposure("AAPL") -> all ETFs that own AAPL, with weights.
    """
    if df is None:
        df = load_etf_holdings()

    asset_symbol = asset_symbol.upper()
    sub = df.loc[df["asset_symbol"] == asset_symbol].copy()
    if "weight_pct" in sub.columns:
        sub = sub.sort_values("weight_pct", ascending=False)
    return sub


__all__ = ["load_etf_holdings", "get_etf_holdings", "get_asset_exposure", "HOLDINGS_PATH"]
