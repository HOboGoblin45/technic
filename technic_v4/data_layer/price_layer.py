# technic_v4/data_layer/price_layer.py

from __future__ import annotations

import time
import json
import asyncio
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Set

import pandas as pd

# We still leverage your existing Polygon client for now.
# This module is the *only* place the rest of Technic should call
# to obtain price / history data.
from technic_v4.data_layer.polygon_client import (
    get_stock_history_df as _polygon_history,
    get_stock_intraday_df as _polygon_intraday,
)


# ------------------------------------------------------------
# Config / tuning
# ------------------------------------------------------------

# How long (seconds) a cached history is considered "fresh"
CACHE_TTL_SECONDS = 60.0

# Simple in-memory cache: (symbol, days, use_intraday) -> (timestamp, df)
_HISTORY_CACHE: Dict[Tuple[str, int, bool], Tuple[float, pd.DataFrame]] = {}

# Realtime last-price store (populated by optional websocket stream)
_REALTIME_LAST: Dict[str, Tuple[float, float]] = {}
_STREAM_THREAD: Optional[threading.Thread] = None
_STREAM_STOP = threading.Event()
_STREAM_SYMBOLS: Set[str] = set()
_STREAM_LOCK = threading.Lock()
_STREAM_APIKEY: Optional[str] = None


def get_stream_status() -> Dict[str, object]:
    """
    Snapshot of the current websocket stream state for UI/debug display.
    """
    status: Dict[str, object] = {
        "active": _STREAM_THREAD is not None and _STREAM_THREAD.is_alive(),
        "symbols": list(_STREAM_SYMBOLS),
        "api_key_set": bool(_STREAM_APIKEY),
    }

    now = _now()
    ages: Dict[str, float] = {}
    for sym, (ts_ms, _price) in _REALTIME_LAST.items():
        ages[sym] = max(0.0, now - (ts_ms / 1000.0))
    status["last_tick_age"] = ages
    return status


@dataclass
class PriceSourceStats:
    source: str
    symbol: str
    days: int
    use_intraday: bool
    bars: int
    latency_ms: float
    from_cache: bool


# ------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------

def _now() -> float:
    return time.time()


def _get_from_cache(symbol: str, days: int, use_intraday: bool) -> Optional[pd.DataFrame]:
    key = (symbol.upper(), int(days), bool(use_intraday))
    hit = _HISTORY_CACHE.get(key)
    if not hit:
        return None

    ts, df = hit
    if _now() - ts > CACHE_TTL_SECONDS:
        # expired
        try:
            del _HISTORY_CACHE[key]
        except KeyError:
            pass
        return None

    # return a shallow copy so callers can't mutate our cache
    return df.copy()


def _store_in_cache(symbol: str, days: int, use_intraday: bool, df: pd.DataFrame) -> None:
    key = (symbol.upper(), int(days), bool(use_intraday))
    _HISTORY_CACHE[key] = (_now(), df.copy())


# ------------------------------------------------------------
# Source attempts (easy to extend later)
# ------------------------------------------------------------

def _try_realtime_window(
    symbol: str,
    days: int,
    use_intraday: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[PriceSourceStats]]:
    """
    Placeholder for a future WebSocket-backed realtime window.

    For now, if a realtime price exists, return a 1-row DataFrame
    so callers can show a live price without waiting for REST.
    """
    sym = symbol.upper()
    last = _REALTIME_LAST.get(sym)
    if not last:
        return None, None

    ts_ms, price = last
    dt_idx = pd.to_datetime([ts_ms], unit="ms", utc=True)
    df = pd.DataFrame({"Close": [price]}, index=dt_idx)

    stats = PriceSourceStats(
        source="realtime_cache",
        symbol=sym,
        days=days,
        use_intraday=use_intraday,
        bars=len(df),
        latency_ms=0.0,
        from_cache=True,
    )
    return df, stats


def _try_polygon_rest(
    symbol: str,
    days: int,
    use_intraday: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[PriceSourceStats]]:
    """
    Call Polygon history.
    - If use_intraday and the window is small, fetch intraday bars
      (5-minute) for a live feel.
    - Otherwise fall back to daily bars.
    """
    t0 = _now()
    df = None
    source = "polygon_rest_daily"

    # Intraday only for short lookbacks to keep responses manageable
    if use_intraday and days <= 5:
        df = _polygon_intraday(symbol=symbol, days=days, multiplier=5, timespan="minute")
        source = "polygon_rest_intraday"

    if df is None:
        try:
            df = _polygon_history(symbol=symbol, days=days)
        except TypeError:
            df = _polygon_history(symbol, days)
        source = "polygon_rest_daily"

    t1 = _now()

    if df is None or df.empty:
        return None, None

    stats = PriceSourceStats(
        source=source,
        symbol=symbol.upper(),
        days=days,
        use_intraday=use_intraday,
        bars=len(df),
        latency_ms=(t1 - t0) * 1000.0,
        from_cache=False,
    )
    return df, stats


def _try_yahoo_fallback(
    symbol: str,
    days: int,
    use_intraday: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[PriceSourceStats]]:
    """
    Optional future Yahoo / other-provider fallback.

    For now, we just return (None, None) so it doesn't affect behavior.
    Later, you can import a Yahoo client here and call it exactly the
    same way as Polygon, with the same return format.
    """
    return None, None


def get_realtime_last(symbol: str) -> Optional[float]:
    """
    Return the latest streamed price if available.
    """
    sym = symbol.upper()
    last = _REALTIME_LAST.get(sym)
    if not last:
        return None
    return float(last[1])


def _stream_loop(symbols: Set[str], api_key: str) -> None:
    """
    Run a websocket loop to listen for trade prints and update _REALTIME_LAST.
    """
    try:
        import websockets  # type: ignore
    except ImportError:
        print("[PRICE STREAM] websockets package not installed; streaming disabled.")
        return

    async def _run_once():
        url = "wss://socket.polygon.io/stocks"
        params = ",".join(f"T.{s}" for s in symbols)
        if not params:
            return

        async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
            await ws.send(json.dumps({"action": "auth", "params": api_key}))
            await ws.send(json.dumps({"action": "subscribe", "params": params}))

            async for raw in ws:
                if _STREAM_STOP.is_set():
                    break
                try:
                    events = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(events, list):
                    events = [events]
                for evt in events:
                    sym = evt.get("sym") or evt.get("pair")
                    price = evt.get("p") or evt.get("ap") or evt.get("c")
                    ts = evt.get("t") or (time.time() * 1000)
                    if sym and price is not None:
                        _REALTIME_LAST[sym.upper()] = (float(ts), float(price))

    while not _STREAM_STOP.is_set():
        try:
            asyncio.run(_run_once())
        except Exception as exc:
            msg = str(exc)
            print(f"[PRICE STREAM] error: {msg}")
            if "policy" in msg.lower() or "1008" in msg:
                # Stop retry loop to avoid log spam when the provider rejects the connection.
                _STREAM_STOP.set()
                break
            time.sleep(2.0)


def start_realtime_stream(symbols: Set[str], api_key: str) -> bool:
    """
    Start (or restart) a background websocket listener for the given symbols.
    Returns True if streaming is active or started; False if prerequisites missing.
    """
    if not symbols or not api_key:
        return False

    with _STREAM_LOCK:
        global _STREAM_THREAD, _STREAM_SYMBOLS, _STREAM_APIKEY, _STREAM_STOP

        need_restart = (
            _STREAM_THREAD is None
            or not _STREAM_THREAD.is_alive()
            or symbols != _STREAM_SYMBOLS
            or api_key != _STREAM_APIKEY
        )

        if not need_restart:
            return True

        # Stop existing thread if running
        if _STREAM_THREAD and _STREAM_THREAD.is_alive():
            _STREAM_STOP.set()
            try:
                _STREAM_THREAD.join(timeout=1.0)
            except Exception:
                pass
            _STREAM_STOP = threading.Event()

        _STREAM_SYMBOLS = set(symbols)
        _STREAM_APIKEY = api_key
        _STREAM_STOP.clear()

        _STREAM_THREAD = threading.Thread(
            target=_stream_loop, args=(_STREAM_SYMBOLS, api_key), daemon=True
        )
        _STREAM_THREAD.start()
        print(f"[PRICE STREAM] started for {len(_STREAM_SYMBOLS)} symbols.")

    return True


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def get_stock_history_df(
    symbol: str,
    days: int,
    use_intraday: bool = True,
) -> pd.DataFrame:
    """
    Unified price/history endpoint for Technic.

    - Checks short-lived cache first
    - Then tries:
        1) Realtime window (WebSocket cache) [stub for now]
        2) Polygon REST (currently daily bars)
        3) Yahoo / other fallback [stub for now]
    - Guarantees a pandas DataFrame on success, or raises RuntimeError.
    """

    symbol = symbol.upper()
    days = int(days)

    # 1) Cache
    cached = _get_from_cache(symbol, days, use_intraday)
    if cached is not None:
        print(f"[PRICE] cache hit: {symbol} days={days} intraday={use_intraday} bars={len(cached)}")
        return cached

    # 2) Realtime window (WebSocket cache) – stub for now
    df, stats = _try_realtime_window(symbol, days, use_intraday)
    if df is not None and not df.empty:
        print(f"[PRICE] realtime window: {stats}")
        _store_in_cache(symbol, days, use_intraday, df)
        return df

    # 3) Polygon REST
    df, stats = _try_polygon_rest(symbol, days, use_intraday)
    if df is not None and not df.empty:
        print(f"[PRICE] polygon_rest: {stats}")
        _store_in_cache(symbol, days, use_intraday, df)
        return df

    # 4) Yahoo / other fallback – stub
    df, stats = _try_yahoo_fallback(symbol, days, use_intraday)
    if df is not None and not df.empty:
        print(f"[PRICE] yahoo_fallback: {stats}")
        _store_in_cache(symbol, days, use_intraday, df)
        return df

    # If we got here, nothing worked
    raise RuntimeError(
        f"PriceLayer could not fetch history for {symbol} (days={days}, intraday={use_intraday})."
    )


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample an OHLCV dataframe to a higher timeframe (e.g., 'W', 'M').
    Assumes columns: Open, High, Low, Close, Volume.
    """
    if df is None or df.empty:
        return df
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    out = df.resample(rule).apply(agg).dropna()
    return out


def get_multi_timeframes(symbol: str, days: int = 365) -> dict[str, pd.DataFrame]:
    """
    Convenience wrapper returning multiple timeframes for a symbol.
    - daily: days window
    - weekly: resampled from daily
    - monthly: resampled from daily
    """
    base = get_stock_history_df(symbol, days=days, use_intraday=False)
    weekly = resample_ohlcv(base, "W")
    monthly = resample_ohlcv(base, "M")
    return {"daily": base, "weekly": weekly, "monthly": monthly}
