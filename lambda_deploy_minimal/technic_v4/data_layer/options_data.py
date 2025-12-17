from __future__ import annotations

import datetime as dt
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

BASE_URL = "https://api.polygon.io"
SNAPSHOT_PAGE_LIMIT = 250
CONTRACTS_PAGE_LIMIT = 1000
MAX_RETRIES = 3
REQUEST_TIMEOUT = 10.0
DEFAULT_SNAPSHOT_TTL = 600.0  # seconds
DEFAULT_CONTRACTS_TTL = 24 * 3600.0


class OptionChainService:
    """
    Lightweight Polygon options client with simple paging + in-memory caching.

    The cache is intentionally short-lived; Streamlit should still wrap calls
    with st.cache_data to persist across reruns.
    """

    def __init__(
        self,
        api_key: str | None = None,
        session: Optional[requests.Session] = None,
        snapshot_ttl: float = DEFAULT_SNAPSHOT_TTL,
        contracts_ttl: float = DEFAULT_CONTRACTS_TTL,
    ) -> None:
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        self.session = session or requests.Session()
        self.snapshot_ttl = float(snapshot_ttl)
        self.contracts_ttl = float(contracts_ttl)

        # cache_key -> (timestamp, results, meta)
        self._snapshot_cache: Dict[
            Tuple[str, Tuple[Tuple[str, Any], ...]], Tuple[float, List[dict], dict]
        ] = {}
        # (symbol, expired) -> (timestamp, results)
        self._contracts_cache: Dict[Tuple[str, bool], Tuple[float, List[dict]]] = {}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[requests.Response]:
        if not self.api_key:
            raise RuntimeError("POLYGON_API_KEY is not set; options data unavailable.")

        params = dict(params or {})
        # Avoid double-adding when next_url already contains apiKey
        if "apiKey" not in params and "apiKey=" not in url:
            params["apiKey"] = self.api_key

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            except requests.Timeout:
                if attempt == MAX_RETRIES:
                    return None
                time.sleep(1.5 * attempt)
                continue

            if resp.status_code == 200:
                return resp

            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt == MAX_RETRIES:
                    return None
                time.sleep(1.5 * attempt)
                continue

            return None

        return None

    def _paginate(self, url: str, params: Optional[Dict[str, Any]] = None) -> Tuple[List[dict], dict]:
        results: List[dict] = []
        meta: dict = {}

        next_url = url
        next_params = params

        while next_url:
            resp = self._get(next_url, next_params)
            if resp is None:
                break

            try:
                payload = resp.json()
            except Exception:
                break

            page_results = payload.get("results") or []
            results.extend(page_results)

            meta = {
                "status": payload.get("status"),
                "count": len(results),
                "next_url": payload.get("next_url"),
            }

            next_url = payload.get("next_url")
            next_params = None  # next_url already includes params

        return results, meta

    def to_dataframe(self, snapshot: Optional[List[dict]]) -> pd.DataFrame:
        """
        Normalize a Polygon options snapshot into a DataFrame with stable, snake_case columns.
        """
        columns = [
            "symbol",
            "option_symbol",
            "expiration_date",
            "side",
            "strike",
            "bid",
            "ask",
            "last",
            "mid_price",
            "bid_ask_spread",
            "bid_ask_spread_pct",
            "volume",
            "open_interest",
            "implied_volatility",
            "delta",
            "gamma",
            "vega",
            "theta",
            "underlying_price",
            "dte",
        ]

        if isinstance(snapshot, dict):
            snapshot = snapshot.get("results") or []
        if not snapshot:
            return pd.DataFrame(columns=columns)

        today = dt.date.today()
        rows: List[Dict[str, Any]] = []

        for item in snapshot:
            details = item.get("details") or {}
            last_quote = item.get("last_quote") or {}
            last_trade = item.get("last_trade") or {}
            day = item.get("day") or {}
            greeks = item.get("greeks") or {}

            option_symbol = details.get("ticker") or item.get("ticker") or item.get("option_symbol")
            expiration_raw = details.get("expiration_date") or item.get("expiration_date")
            side = details.get("contract_type") or item.get("contract_type") or item.get("side")
            strike = details.get("strike_price") or item.get("strike")

            bid = last_quote.get("bid")
            ask = last_quote.get("ask")
            last = last_trade.get("price") or day.get("close") or item.get("last")

            underlying_info = details.get("underlying_asset") or item.get("underlying_asset") or {}
            underlying_price = underlying_info.get("price") if isinstance(underlying_info, dict) else None
            symbol = None
            if isinstance(underlying_info, dict):
                symbol = underlying_info.get("ticker")
            symbol = symbol or item.get("underlying_symbol") or item.get("sym")

            volume = day.get("volume") or item.get("volume") or last_trade.get("volume")
            open_interest = item.get("open_interest")
            iv = item.get("implied_volatility") or item.get("iv")

            delta = greeks.get("delta")
            gamma = greeks.get("gamma")
            vega = greeks.get("vega")
            theta = greeks.get("theta")

            mid_price = None
            bid_ask_spread = None
            bid_ask_spread_pct = None
            if bid is not None and ask is not None:
                mid_price = (bid + ask) / 2.0
                bid_ask_spread = ask - bid
                if mid_price:
                    bid_ask_spread_pct = bid_ask_spread / mid_price

            dte = None
            if expiration_raw:
                try:
                    exp_date = pd.to_datetime(expiration_raw).date()
                    dte = (exp_date - today).days
                    expiration_raw = exp_date
                except Exception:
                    pass

            row = {
                "symbol": symbol,
                "option_symbol": option_symbol,
                "expiration_date": expiration_raw,
                "side": side,
                "strike": strike,
                "bid": bid,
                "ask": ask,
                "last": last,
                "mid_price": mid_price,
                "bid_ask_spread": bid_ask_spread,
                "bid_ask_spread_pct": bid_ask_spread_pct,
                "volume": volume,
                "open_interest": open_interest,
                "implied_volatility": iv,
                "delta": delta,
                "gamma": gamma,
                "vega": vega,
                "theta": theta,
                "underlying_price": underlying_price,
                "dte": dte,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        return df[columns] if not df.empty else pd.DataFrame(columns=columns)

    def get_chain(
        self,
        symbol: str,
        contract_type: str | None = None,
        expiration_date: str | None = None,
        expiration_date_to: str | None = None,
        strike_price_gte: float | None = None,
        strike_price_lte: float | None = None,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Convenience wrapper: fetch chain snapshot and return a normalized DataFrame + meta.
        """
        contracts, meta = self.fetch_chain_snapshot(
            symbol=symbol,
            contract_type=contract_type,
            expiration_date=expiration_date,
            expiration_date_to=expiration_date_to,
            strike_price_gte=strike_price_gte,
            strike_price_lte=strike_price_lte,
        )
        try:
            df = self.to_dataframe(contracts)
        except Exception:
            df = pd.DataFrame()
        return df, meta

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fetch_chain_snapshot(
        self,
        symbol: str,
        contract_type: str | None = None,
        expiration_date: str | None = None,
        expiration_date_to: str | None = None,
        strike_price_gte: float | None = None,
        strike_price_lte: float | None = None,
    ) -> Tuple[List[dict], dict]:
        """
        Pull the snapshot chain for an underlying, following pagination when needed.
        Returns (contracts, meta).
        """
        sym = symbol.upper().strip()
        filters: Dict[str, Any] = {"limit": SNAPSHOT_PAGE_LIMIT}

        if contract_type:
            filters["contract_type"] = contract_type.lower()
        if expiration_date:
            filters["expiration_date"] = expiration_date
        if expiration_date_to:
            filters["expiration_date.lte"] = expiration_date_to
        if strike_price_gte is not None:
            filters["strike_price.gte"] = strike_price_gte
        if strike_price_lte is not None:
            filters["strike_price.lte"] = strike_price_lte

        cache_key = (sym, tuple(sorted((k, v) for k, v in filters.items() if k != "limit")))
        now = time.time()
        cached = self._snapshot_cache.get(cache_key)
        if cached and (now - cached[0] <= self.snapshot_ttl):
            ts, cached_results, cached_meta = cached
            meta = dict(cached_meta)
            meta["cached"] = True
            meta["fetched_at"] = ts
            return cached_results, meta

        url = f"{BASE_URL}/v3/snapshot/options/{sym}"
        results, meta = self._paginate(url, filters)
        meta["cached"] = False
        meta["fetched_at"] = now

        self._snapshot_cache[cache_key] = (now, results, meta.copy())
        return results, meta

    def fetch_contracts_list(self, symbol: str, expired: bool = False) -> List[dict]:
        """
        Pull reference contract metadata for an underlying (without greeks/prices).
        Cached for a day since the list changes slowly.
        """
        sym = symbol.upper().strip()
        cache_key = (sym, bool(expired))
        now = time.time()
        cached = self._contracts_cache.get(cache_key)
        if cached and (now - cached[0] <= self.contracts_ttl):
            return cached[1]

        params = {
            "underlying_ticker": sym,
            "limit": CONTRACTS_PAGE_LIMIT,
            "expired": str(bool(expired)).lower(),
        }
        url = f"{BASE_URL}/v3/reference/options/contracts"
        results, _meta = self._paginate(url, params)
        self._contracts_cache[cache_key] = (now, results)
        return results

    def fetch_option_history(
        self,
        option_ticker: str,
        days: int = 90,
        multiplier: int = 1,
        timespan: str = "day",
    ) -> List[dict]:
        """
        Fetch historical bars for a single option contract.
        Returns a list of dicts with ts, open, high, low, close, volume.
        """
        if days <= 0:
            return []

        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=days + 5)

        path = (
            f"/v2/aggs/ticker/{option_ticker}/range/"
            f"{multiplier}/{timespan}/{start_date}/{end_date}"
        )
        resp = self._get(f"{BASE_URL}{path}", params={"adjusted": "true", "sort": "asc", "limit": 5000})
        if resp is None:
            return []

        try:
            payload = resp.json()
        except Exception:
            return []

        bars = []
        for bar in payload.get("results") or []:
            ts = bar.get("t")
            if ts is None:
                continue
            bars.append(
                {
                    "timestamp": ts,
                    "open": bar.get("o"),
                    "high": bar.get("h"),
                    "low": bar.get("l"),
                    "close": bar.get("c"),
                    "volume": bar.get("v"),
                }
            )

        return bars
