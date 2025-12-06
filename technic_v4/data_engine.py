"""Unified data access facade so engine code is source-agnostic."""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from technic_v4.data_layer.market_cache import MarketCache
from technic_v4.data_layer.price_layer import get_stock_history_df as _price_history
from technic_v4.data_layer.fundamentals import get_fundamentals as _fundamentals
from technic_v4.data_layer.options_data import OptionChainService


_MARKET_CACHE: Optional[MarketCache] = None
_OPTION_SERVICE: Optional[OptionChainService] = None


def _get_market_cache() -> Optional[MarketCache]:
    global _MARKET_CACHE
    if _MARKET_CACHE is None:
        try:
            _MARKET_CACHE = MarketCache()
        except Exception as exc:
            print(f"[data_engine] MarketCache unavailable: {exc}")
            _MARKET_CACHE = None
    return _MARKET_CACHE


def _ensure_option_service() -> OptionChainService:
    global _OPTION_SERVICE
    if _OPTION_SERVICE is None:
        _OPTION_SERVICE = OptionChainService()
    return _OPTION_SERVICE


def _standardize_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    out = df.copy()
    # Normalize column names
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    out = out.rename(columns=rename_map)

    if "Date" not in out.columns:
        if out.index.name == "Date" or isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={"index": "Date"})
        elif "timestamp" in out.columns:
            out = out.rename(columns={"timestamp": "Date"})

    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"])
        out = out.set_index("Date")

    cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in cols if c not in out.columns]
    if missing:
        print(f"[data_engine] missing columns {missing}")
    return out[cols] if all(c in out.columns for c in cols) else out


def get_price_history(symbol: str, days: int, freq: str = "daily") -> pd.DataFrame:
    """Fetch price history with cache-first logic and Polygon fallback."""
    symbol = symbol.upper().strip()
    if days <= 0:
        return pd.DataFrame()

    try:
        if freq == "daily":
            cache = _get_market_cache()
            if cache:
                try:
                    df = cache.get_symbol_history(symbol, days)
                    if df is not None and not df.empty and len(df) >= days:
                        return _standardize_history(df.tail(days))
                except Exception as exc:
                    print(f"[data_engine] MarketCache miss for {symbol}: {exc}")

            try:
                df = _price_history(symbol=symbol, days=days, use_intraday=False)
                return _standardize_history(df)
            except Exception as exc:
                print(f"[data_engine] Polygon daily failed for {symbol}: {exc}")
                return pd.DataFrame()

        # Intraday or other frequencies
        try:
            df = _price_history(symbol=symbol, days=days, use_intraday=True)
            return _standardize_history(df)
        except Exception as exc:
            print(f"[data_engine] intraday fetch failed for {symbol}: {exc}")
            return pd.DataFrame()

    except Exception as exc:
        print(f"[data_engine] unexpected price history error for {symbol}: {exc}")
        return pd.DataFrame()


def get_fundamentals(symbol: str, as_of_date: Optional[date] = None):
    """Return fundamentals snapshot (latest)."""
    try:
        return _fundamentals(symbol)
    except Exception as exc:
        print(f"[data_engine] fundamentals error for {symbol}: {exc}")
        return None


def get_options_chain(symbol: str, as_of_date: Optional[date] = None) -> pd.DataFrame:
    """Return options chain DataFrame; best-effort."""
    try:
        service = _ensure_option_service()
        contracts, _meta = service.fetch_chain_snapshot(symbol)
        return pd.DataFrame(contracts)
    except Exception as exc:
        print(f"[data_engine] options chain error for {symbol}: {exc}")
        return pd.DataFrame()


__all__ = ["get_price_history", "get_fundamentals", "get_options_chain"]
