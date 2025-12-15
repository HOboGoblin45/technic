#!/usr/bin/env python3
"""
Week 2 Scanner Optimizations - Push to 60 Seconds
==================================================

Current: 75-90s for 5,000-6,000 tickers (0.015s/symbol)
Target: 60s for 5,000-6,000 tickers (0.010-0.012s/symbol)

Optimizations:
1. Increased Ray workers (20 → 50)
2. Async I/O for API calls
3. Enhanced pre-screening
4. Ray object store for shared data
"""

import asyncio
import aiohttp
from typing import List, Dict, Any
import time


# ============================================================================
# OPTIMIZATION 1: Async API Calls
# ============================================================================

async def fetch_price_async(session: aiohttp.ClientSession, symbol: str, api_key: str) -> Dict[str, Any]:
    """
    Async API call for price data - non-blocking I/O
    """
    from datetime import datetime, timedelta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=150)
    
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
        f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        f"?adjusted=true&sort=asc&apiKey={api_key}"
    )
    
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status == 200:
                data = await response.json()
                return {"symbol": symbol, "data": data, "error": None}
            else:
                return {"symbol": symbol, "data": None, "error": f"HTTP {response.status}"}
    except Exception as e:
        return {"symbol": symbol, "data": None, "error": str(e)}


async def batch_fetch_async(symbols: List[str], api_key: str, batch_size: int = 50) -> List[Dict[str, Any]]:
    """
    Fetch multiple symbols concurrently with batching
    
    Args:
        symbols: List of ticker symbols
        api_key: Polygon API key
        batch_size: Number of concurrent requests (default 50)
    
    Returns:
        List of results with price data
    """
    results = []
    
    # Process in batches to avoid overwhelming the API
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_price_async(session, sym, api_key) for sym in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        
        # Small delay between batches to respect rate limits
        if i + batch_size < len(symbols):
            await asyncio.sleep(0.1)
    
    return results


def get_prices_async_wrapper(symbols: List[str], api_key: str) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for async price fetching
    """
    return asyncio.run(batch_fetch_async(symbols, api_key))


# ============================================================================
# OPTIMIZATION 2: Enhanced Pre-screening
# ============================================================================

# Low-volume ticker blacklist (updated periodically)
LOW_VOLUME_BLACKLIST = {
    # Penny stocks and low-volume tickers
    'TOPS', 'SHIP', 'CTRM', 'SNDL', 'BNGO', 'PLUG', 'FCEL', 'BLNK',
    # Add more as identified
}

def enhanced_pre_screen(symbol: str, market_cap: float, sector: str, avg_volume: float) -> bool:
    """
    Enhanced pre-screening before expensive data fetch
    
    Filters:
    1. Market cap minimum ($500M)
    2. Sector focus (liquid sectors)
    3. Volume minimum (1M shares/day)
    4. Blacklist check
    
    Returns:
        True if symbol should be scanned, False to skip
    """
    # Filter 1: Market cap minimum
    if market_cap and market_cap < 500_000_000:  # $500M minimum
        return False
    
    # Filter 2: Sector focus (liquid sectors only)
    LIQUID_SECTORS = {
        'Technology', 'Healthcare', 'Financials', 'Energy', 
        'Industrials', 'Consumer Cyclical', 'Communication Services'
    }
    if sector and sector not in LIQUID_SECTORS:
        return False
    
    # Filter 3: Volume minimum
    if avg_volume and avg_volume < 1_000_000:  # 1M shares/day minimum
        return False
    
    # Filter 4: Blacklist check
    if symbol in LOW_VOLUME_BLACKLIST:
        return False
    
    return True


# ============================================================================
# OPTIMIZATION 3: Ray Object Store for Shared Data
# ============================================================================

import ray

@ray.remote
class SharedDataStore:
    """
    Ray actor for shared data across workers
    Stores frequently accessed data in Ray's object store
    """
    
    def __init__(self):
        self.market_data = {}
        self.regime_data = {}
        self.macro_data = {}
    
    def set_market_data(self, data: Dict[str, Any]):
        """Store market regime data"""
        self.market_data = data
    
    def get_market_data(self) -> Dict[str, Any]:
        """Retrieve market regime data"""
        return self.market_data
    
    def set_macro_data(self, data: Dict[str, Any]):
        """Store macro indicators"""
        self.macro_data = data
    
    def get_macro_data(self) -> Dict[str, Any]:
        """Retrieve macro indicators"""
        return self.macro_data


def init_shared_store():
    """
    Initialize Ray shared data store
    """
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    store = SharedDataStore.remote()
    return store


# ============================================================================
# OPTIMIZATION 4: Incremental Scanning
# ============================================================================

from datetime import datetime, timedelta
import pandas as pd

def get_symbols_to_scan(
    universe: List[str],
    last_scan_time: datetime = None,
    cache_max_age_hours: int = 1
) -> List[str]:
    """
    Intelligent symbol selection for incremental scanning
    
    Strategy:
    1. Always scan: High-priority symbols (top 500 by volume)
    2. Changed symbols: Symbols with recent news/events
    3. Cached symbols: Random sample if cache is fresh
    
    Args:
        universe: Full universe of symbols
        last_scan_time: When the last scan completed
        cache_max_age_hours: Max age of cached results to use
    
    Returns:
        List of symbols to scan
    """
    if last_scan_time is None:
        # First scan - scan everything
        return universe
    
    # Check if cache is too old
    cache_age = datetime.now() - last_scan_time
    if cache_age > timedelta(hours=cache_max_age_hours):
        # Cache expired - full scan
        return universe
    
    # Incremental scan strategy
    symbols_to_scan = []
    
    # 1. High-priority symbols (always scan)
    high_priority = get_high_priority_symbols(universe, top_n=500)
    symbols_to_scan.extend(high_priority)
    
    # 2. Changed symbols (recent activity)
    changed = get_changed_symbols(universe, since=last_scan_time)
    symbols_to_scan.extend(changed)
    
    # 3. Random sample of rest (for diversity)
    remaining = [s for s in universe if s not in symbols_to_scan]
    import random
    sample_size = min(500, len(remaining))
    symbols_to_scan.extend(random.sample(remaining, k=sample_size))
    
    return list(set(symbols_to_scan))  # Remove duplicates


def get_high_priority_symbols(universe: List[str], top_n: int = 500) -> List[str]:
    """
    Get high-priority symbols (high volume, high market cap)
    """
    # TODO: Implement based on universe metadata
    # For now, return first N symbols
    return universe[:top_n]


def get_changed_symbols(universe: List[str], since: datetime) -> List[str]:
    """
    Get symbols with recent changes (news, earnings, etc.)
    """
    # TODO: Implement based on event data
