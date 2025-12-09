"""Unified data access facade so engine code is source-agnostic."""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
import requests
from technic_v4.config import POLYGON_API_KEY

from technic_v4.data_layer.market_cache import MarketCache
from technic_v4.data_layer.price_layer import get_stock_history_df as _price_history
from technic_v4.data_layer.fundamentals import get_fundamentals as _fundamentals
from technic_v4.data_layer.options_data import OptionChainService
from technic_v4.infra.logging import get_logger

logger = get_logger()


_MARKET_CACHE: Optional[MarketCache] = None
_OPTION_SERVICE: Optional[OptionChainService] = None


def _get_market_cache() -> Optional[MarketCache]:
    global _MARKET_CACHE
    if _MARKET_CACHE is None:
        try:
            _MARKET_CACHE = MarketCache()
        except Exception as exc:
            logger.warning("[data_engine] MarketCache unavailable: %s", exc)
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
        logger.warning("[data_engine] missing columns %s", missing)
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
                        logger.info("[data_engine] cache hit for %s (%d bars)", symbol, len(df))
                        return _standardize_history(df.tail(days))
                except Exception as exc:
                    logger.warning("[data_engine] MarketCache miss for %s: %s", symbol, exc)

            try:
                df = _price_history(symbol=symbol, days=days, use_intraday=False)
                logger.info("[data_engine] fallback to Polygon daily for %s", symbol)
                return _standardize_history(df)
            except Exception as exc:
                logger.error("[data_engine] Polygon daily failed for %s", symbol, exc_info=True)
                return pd.DataFrame()

        # Intraday or other frequencies
        try:
            df = _price_history(symbol=symbol, days=days, use_intraday=True)
            logger.info("[data_engine] intraday fetch for %s", symbol)
            return _standardize_history(df)
        except Exception as exc:
            logger.error("[data_engine] intraday fetch failed for %s", symbol, exc_info=True)
            return pd.DataFrame()

    except Exception as exc:
        logger.error("[data_engine] unexpected price history error for %s", symbol, exc_info=True)
        return pd.DataFrame()

def get_ticker_details(symbol: str) -> dict:
    """
    Fetch Polygon ticker details including market_cap.
    Returns the raw 'results' dict or {} on failure.
    """
    api_key = POLYGON_API_KEY
    if not api_key:
        # No key configured; skip network call
        return {}

    sym = (symbol or "").upper().strip()
    if not sym:
        return {}

    try:
        url = "https://api.polygon.io/v3/reference/tickers/" + sym
        params = {"apiKey": api_key}
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json() or {}
        return data.get("results") or {}
    except Exception as exc:
        logger.warning("[data_engine] ticker details failed for %s: %s", sym, exc)
        return {}

def get_fundamentals(symbol: str, as_of_date: Optional[date] = None):
    """Return fundamentals snapshot (latest)."""
    try:
        return _fundamentals(symbol)
    except Exception as exc:
        logger.error("[data_engine] fundamentals error for %s", symbol, exc_info=True)
        return None


def get_options_chain(symbol: str, as_of_date: Optional[date] = None) -> pd.DataFrame:
    """Return options chain DataFrame; best-effort."""
    try:
        service = _ensure_option_service()
        contracts, _meta = service.fetch_chain_snapshot(symbol)
        return pd.DataFrame(contracts)
    except Exception as exc:
        logger.error("[data_engine] options chain error for %s", symbol, exc_info=True)
        return pd.DataFrame()


__all__ = ["get_price_history", "get_fundamentals", "get_options_chain", "get_ticker_details"]
