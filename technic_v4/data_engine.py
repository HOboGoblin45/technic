"""Unified data access facade so engine code is source-agnostic."""

from __future__ import annotations

from datetime import date
from typing import Optional
import time

import pandas as pd
import requests
import os

from technic_v4.data_layer.market_cache import MarketCache
from technic_v4.data_layer.price_layer import get_stock_history_df as _price_history
from technic_v4.data_layer.fundamentals import get_fundamentals as _fundamentals
from technic_v4.data_layer.options_data import OptionChainService
from technic_v4.infra.logging import get_logger

# Optional Redis import (graceful degradation)
try:
    import redis
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    RedisError = Exception  # Fallback

logger = get_logger()

# L3 Redis cache (distributed, cross-process)
_REDIS_CLIENT = None
_REDIS_ENABLED = False

def _init_redis():
    """Initialize Redis client (best-effort)"""
    global _REDIS_CLIENT, _REDIS_ENABLED
    
    if not REDIS_AVAILABLE:
        _REDIS_ENABLED = False
        logger.info("[REDIS] Redis module not installed, using L1/L2 cache only")
        return
    
    try:
        settings = get_settings()
        redis_url = getattr(settings, 'redis_url', 'redis://localhost:6379/0')
        _REDIS_CLIENT = redis.from_url(redis_url, decode_responses=False, socket_timeout=2)
        _REDIS_CLIENT.ping()
        _REDIS_ENABLED = True
        logger.info("[REDIS] Connected to Redis at %s", redis_url)
    except Exception as e:
        _REDIS_ENABLED = False
        logger.warning("[REDIS] Redis unavailable, using L1/L2 cache only: %s", e)

def get_redis_client():
    """Get Redis client (lazy init)"""
    global _REDIS_CLIENT, _REDIS_ENABLED
    if _REDIS_CLIENT is None:
        _init_redis()
    return _REDIS_CLIENT if _REDIS_ENABLED else None


_MARKET_CACHE: Optional[MarketCache] = None
_OPTION_SERVICE: Optional[OptionChainService] = None

# In-memory cache for ultra-fast repeated access (L1 cache)
_MEMORY_CACHE = {}
_CACHE_TTL = 3600  # 1 hour TTL for in-memory cache
_CACHE_STATS = {"hits": 0, "misses": 0, "total": 0}


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
    """Fetch price history with multi-layer cache (memory → MarketCache → Polygon)."""
    symbol = symbol.upper().strip()
    if days <= 0:
        return pd.DataFrame()

    # Track cache statistics
    _CACHE_STATS["total"] += 1

    # L1 Cache: Check in-memory cache first (fastest)
    cache_key = f"{symbol}_{days}_{freq}"
    now = time.time()
    
    if cache_key in _MEMORY_CACHE:
        cached_data, timestamp = _MEMORY_CACHE[cache_key]
        if now - timestamp < _CACHE_TTL:
            _CACHE_STATS["hits"] += 1
            hit_rate = (_CACHE_STATS["hits"] / _CACHE_STATS["total"]) * 100
            logger.debug("[data_engine] L1 cache hit for %s (hit rate: %.1f%%)", symbol, hit_rate)
            return cached_data.copy()  # Return copy to avoid mutations
    
    _CACHE_STATS["misses"] += 1

    try:
        if freq == "daily":
            # L2 Cache: Try MarketCache (persistent cache)
            cache = _get_market_cache()
            if cache:
                try:
                    df = cache.get_symbol_history(symbol, days)
                    if df is not None and not df.empty and len(df) >= days:
                        logger.info("[data_engine] L2 cache hit for %s (%d bars)", symbol, len(df))
                        result = _standardize_history(df.tail(days))
                        # Promote to L1 cache
                        _MEMORY_CACHE[cache_key] = (result.copy(), now)
                        return result
                except Exception as exc:
                    logger.warning("[data_engine] MarketCache miss for %s: %s", symbol, exc)

            # L3: Fetch from Polygon API (slowest)
            try:
                df = _price_history(symbol=symbol, days=days, use_intraday=False)
                logger.info("[data_engine] Polygon API fetch for %s", symbol)
                result = _standardize_history(df)
                # Store in L1 cache for future requests
                _MEMORY_CACHE[cache_key] = (result.copy(), now)
                return result
            except Exception as exc:
                logger.error("[data_engine] Polygon daily failed for %s", symbol, exc_info=True)
                return pd.DataFrame()

        # Intraday or other frequencies
        try:
            df = _price_history(symbol=symbol, days=days, use_intraday=True)
            logger.info("[data_engine] intraday fetch for %s", symbol)
            result = _standardize_history(df)
            # Store in L1 cache
            _MEMORY_CACHE[cache_key] = (result.copy(), now)
            return result
        except Exception as exc:
            logger.error("[data_engine] intraday fetch failed for %s", symbol, exc_info=True)
            return pd.DataFrame()

    except Exception as exc:
        logger.error("[data_engine] unexpected price history error for %s", symbol, exc_info=True)
        return pd.DataFrame()


def clear_redis_cache():
    """Clear Redis cache (best-effort)"""
    redis_client = get_redis_client()
    if redis_client:
        try:
            redis_client.flushdb()
            logger.info("[REDIS] Redis cache cleared")
        except Exception as e:
            logger.warning("[REDIS] Failed to clear cache: %s", e)

def clear_memory_cache():
    """Clear the in-memory cache (useful for testing or forcing fresh data)."""
    global _MEMORY_CACHE, _CACHE_STATS
    _MEMORY_CACHE.clear()
    _CACHE_STATS = {"hits": 0, "misses": 0, "total": 0}
    logger.info("[data_engine] Memory cache cleared")


def get_cache_stats() -> dict:
    """Return cache performance statistics."""
    total = _CACHE_STATS["total"]
    if total == 0:
        return {"hits": 0, "misses": 0, "total": 0, "hit_rate": 0.0}
    
    hit_rate = (_CACHE_STATS["hits"] / total) * 100
    return {
        "hits": _CACHE_STATS["hits"],
        "misses": _CACHE_STATS["misses"],
        "total": total,
        "hit_rate": hit_rate,
        "cache_size": len(_MEMORY_CACHE)
    }

def get_ticker_details(symbol: str) -> dict:
    """
    Fetch Polygon ticker details including market_cap.
    Returns the raw 'results' dict or {} on failure.
    """
    api_key = os.getenv("POLYGON_API_KEY")
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


__all__ = [
    "get_price_history",
    "get_fundamentals",
    "get_options_chain",
    "get_ticker_details",
    "clear_memory_cache",
    "get_cache_stats"
]
