"""
Smart Cache Warming Engine
Implements multiple warming strategies to achieve >85% cache hit rate
Path 1 Task 6: Smart Cache Warming
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from technic_v4.cache.access_tracker import AccessPatternTracker, get_tracker
from technic_v4.cache.redis_cache import redis_cache
from technic_v4 import data_engine


logger = logging.getLogger(__name__)


class WarmingStrategy(Enum):
    """Cache warming strategies"""
    POPULAR = "popular"  # Most accessed symbols
    MARKET_HOURS = "market_hours"  # Pre-warm before market open
    SECTOR_ROTATION = "sector_rotation"  # Trending sectors
    PREDICTIVE = "predictive"  # ML-based prediction
    TIME_BASED = "time_based"  # Time-of-day patterns


@dataclass
class WarmingConfig:
    """Configuration for cache warming"""
    enabled: bool = True
    strategies: Dict[str, Dict] = None
    refresh_threshold: float = 0.8  # Refresh at 80% TTL
    max_concurrent: int = 10
    rate_limit: int = 100  # per minute
    memory_limit_mb: int = 500
    
    def __post_init__(self):
        if self.strategies is None:
            self.strategies = {
                'popular': {
                    'enabled': True,
                    'limit': 100,
                    'interval': 1800,  # 30 minutes
                },
                'market_hours': {
                    'enabled': True,
                    'pre_warm_time': '08:30',
                    'symbols': 200,
                },
                'sector_rotation': {
                    'enabled': True,
                    'interval': 3600,  # 1 hour
                    'sectors': ['Technology', 'Healthcare', 'Finance'],
                },
                'predictive': {
                    'enabled': True,
                    'confidence_threshold': 0.7,
                },
                'time_based': {
                    'enabled': True,
                    'look_ahead_hours': 1,
                }
            }


@dataclass
class WarmingResult:
    """Result of a warming operation"""
    strategy: str
    symbols_warmed: int
    symbols_failed: int
    duration_seconds: float
    cache_hits_added: int
    errors: List[str] = None


class SmartCacheWarmer:
    """
    Intelligent cache warming with multiple strategies
    
    Features:
    - Popular symbol warming
    - Time-based predictive warming
    - Sector rotation warming
    - Background refresh
    - Rate limiting
    - Resource management
    
    Example:
        >>> warmer = SmartCacheWarmer()
        >>> result = warmer.warm_popular_symbols(limit=100)
        >>> print(f"Warmed {result.symbols_warmed} symbols")
    """
    
    def __init__(
        self,
        config: Optional[WarmingConfig] = None,
        tracker: Optional[AccessPatternTracker] = None
    ):
        """
        Initialize cache warmer
        
        Args:
            config: Warming configuration
            tracker: Access pattern tracker
        """
        self.config = config or WarmingConfig()
        self.tracker = tracker or get_tracker()
        
        # Warming state
        self.last_warming = {}  # strategy -> timestamp
        self.warming_queue = set()  # symbols currently being warmed
        self.warming_stats = {
            'total_warmed': 0,
            'total_failed': 0,
            'by_strategy': {}
        }
        
        # Rate limiting
        self.api_calls_minute = []
        self.last_rate_check = time.time()
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = time.time()
        
        # Clean old calls
        cutoff = now - 60
        self.api_calls_minute = [t for t in self.api_calls_minute if t > cutoff]
        
        # Check limit
        if len(self.api_calls_minute) >= self.config.rate_limit:
            return False
        
        self.api_calls_minute.append(now)
        return True
    
    def _should_warm_strategy(self, strategy: str) -> bool:
        """Check if strategy should run based on interval"""
        if strategy not in self.config.strategies:
            return False
        
        strategy_config = self.config.strategies[strategy]
        if not strategy_config.get('enabled', True):
            return False
        
        interval = strategy_config.get('interval')
        if not interval:
            return True
        
        last_run = self.last_warming.get(strategy, 0)
        return (time.time() - last_run) >= interval
    
    async def _warm_symbol(self, symbol: str, days: int = 60) -> bool:
        """
        Warm cache for a single symbol
        
        Args:
            symbol: Stock symbol
            days: Lookback days
        
        Returns:
            True if successful
        """
        if not self._check_rate_limit():
            await asyncio.sleep(1)  # Wait for rate limit
            return False
        
        try:
            # Fetch and cache price data
            data = data_engine.get_price_history(
                symbol=symbol,
                days=days,
                freq="daily"
            )
            
            if data is not None and not data.empty:
                # Track the access for pattern learning
                self.tracker.track_access(symbol, context={'warmed': True})
                return True
            
            return False
        
        except Exception as e:
            logger.warning(f"Failed to warm {symbol}: {e}")
            return False
    
    async def _warm_symbols_batch(
        self,
        symbols: List[str],
        days: int = 60
    ) -> Tuple[int, int]:
        """
        Warm multiple symbols in parallel
        
        Args:
            symbols: List of symbols
            days: Lookback days
        
        Returns:
            Tuple of (successful, failed) counts
        """
        # Remove duplicates and already warming
        symbols = list(set(symbols) - self.warming_queue)
        
        # Add to warming queue
        self.warming_queue.update(symbols)
        
        try:
            # Warm in batches to respect concurrency limit
            batch_size = self.config.max_concurrent
            successful = 0
            failed = 0
            
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                
                # Warm batch concurrently
                tasks = [self._warm_symbol(sym, days) for sym in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Count results
                for result in results:
                    if isinstance(result, bool) and result:
                        successful += 1
                    else:
                        failed += 1
            
            return successful, failed
        
        finally:
            # Remove from warming queue
            self.warming_queue.difference_update(symbols)
    
    def warm_popular_symbols(
        self,
        limit: Optional[int] = None,
        days: int = 60
    ) -> WarmingResult:
        """
        Warm most popular symbols
        
        Args:
            limit: Number of symbols (default: from config)
            days: Lookback days
        
        Returns:
            WarmingResult with statistics
        """
        start_time = time.time()
        strategy = 'popular'
        
        # Get limit from config if not specified
        if limit is None:
            limit = self.config.strategies[strategy].get('limit', 100)
        
        # Get popular symbols
        symbols = self.tracker.get_popular_symbols(limit=limit)
        
        if not symbols:
            return WarmingResult(
                strategy=strategy,
                symbols_warmed=0,
                symbols_failed=0,
                duration_seconds=0,
                cache_hits_added=0
            )
        
        # Warm symbols
        successful, failed = asyncio.run(self._warm_symbols_batch(symbols, days))
        
        # Update stats
        duration = time.time() - start_time
        self.last_warming[strategy] = time.time()
        self.warming_stats['total_warmed'] += successful
        self.warming_stats['total_failed'] += failed
        
        if strategy not in self.warming_stats['by_strategy']:
            self.warming_stats['by_strategy'][strategy] = {'warmed': 0, 'failed': 0}
        self.warming_stats['by_strategy'][strategy]['warmed'] += successful
        self.warming_stats['by_strategy'][strategy]['failed'] += failed
        
        logger.info(
            f"[WARMING] Popular: {successful}/{len(symbols)} symbols in {duration:.1f}s"
        )
        
        return WarmingResult(
            strategy=strategy,
            symbols_warmed=successful,
            symbols_failed=failed,
            duration_seconds=duration,
            cache_hits_added=successful
        )
    
    def warm_by_time_pattern(
        self,
        look_ahead_hours: int = 1,
        limit: int = 50
    ) -> WarmingResult:
        """
        Warm symbols based on time-of-day patterns
        
        Args:
            look_ahead_hours: Hours to look ahead
            limit: Maximum symbols
        
        Returns:
            WarmingResult
        """
        start_time = time.time()
        strategy = 'time_based'
        
        # Get current hour and look-ahead hour
        current_hour = datetime.now().hour
        target_hour = (current_hour + look_ahead_hours) % 24
        
        # Get symbols popular at target hour
        symbols = self.tracker.get_popular_by_time(target_hour, limit=limit)
        
        if not symbols:
            return WarmingResult(
                strategy=strategy,
                symbols_warmed=0,
                symbols_failed=0,
                duration_seconds=0,
                cache_hits_added=0
            )
        
        # Warm symbols
        successful, failed = asyncio.run(self._warm_symbols_batch(symbols))
        
        # Update stats
        duration = time.time() - start_time
        self.last_warming[strategy] = time.time()
        self.warming_stats['total_warmed'] += successful
        self.warming_stats['total_failed'] += failed
        
        logger.info(
            f"[WARMING] Time-based (hour {target_hour}): "
            f"{successful}/{len(symbols)} symbols in {duration:.1f}s"
        )
        
        return WarmingResult(
            strategy=strategy,
            symbols_warmed=successful,
            symbols_failed=failed,
            duration_seconds=duration,
            cache_hits_added=successful
        )
    
    def warm_trending(
        self,
        window_hours: int = 24,
        limit: int = 50
    ) -> WarmingResult:
        """
        Warm trending symbols
        
        Args:
            window_hours: Time window for trending
            limit: Maximum symbols
        
        Returns:
            WarmingResult
        """
        start_time = time.time()
        strategy = 'sector_rotation'
        
        # Get trending symbols
        symbols = self.tracker.get_trending_symbols(
            window_hours=window_hours,
            limit=limit
        )
        
        if not symbols:
            return WarmingResult(
                strategy=strategy,
                symbols_warmed=0,
                symbols_failed=0,
                duration_seconds=0,
                cache_hits_added=0
            )
        
        # Warm symbols
        successful, failed = asyncio.run(self._warm_symbols_batch(symbols))
        
        # Update stats
        duration = time.time() - start_time
        self.last_warming[strategy] = time.time()
        self.warming_stats['total_warmed'] += successful
        self.warming_stats['total_failed'] += failed
        
        logger.info(
            f"[WARMING] Trending: {successful}/{len(symbols)} symbols in {duration:.1f}s"
        )
        
        return WarmingResult(
            strategy=strategy,
            symbols_warmed=successful,
            symbols_failed=failed,
            duration_seconds=duration,
            cache_hits_added=successful
        )
    
    def warm_predictive(
        self,
        current_symbols: List[str],
        limit: int = 20
    ) -> WarmingResult:
        """
        Predictive warming based on access patterns
        
        Args:
            current_symbols: Currently accessed symbols
            limit: Maximum predictions
        
        Returns:
            WarmingResult
        """
        start_time = time.time()
        strategy = 'predictive'
        
        # Get predicted symbols
        symbols = self.tracker.predict_next_symbols(current_symbols, limit=limit)
        
        if not symbols:
            return WarmingResult(
                strategy=strategy,
                symbols_warmed=0,
                symbols_failed=0,
                duration_seconds=0,
                cache_hits_added=0
            )
        
        # Warm symbols
        successful, failed = asyncio.run(self._warm_symbols_batch(symbols))
        
        # Update stats
        duration = time.time() - start_time
        self.last_warming[strategy] = time.time()
        self.warming_stats['total_warmed'] += successful
        self.warming_stats['total_failed'] += failed
        
        logger.info(
            f"[WARMING] Predictive: {successful}/{len(symbols)} symbols in {duration:.1f}s"
        )
        
        return WarmingResult(
            strategy=strategy,
            symbols_warmed=successful,
            symbols_failed=failed,
            duration_seconds=duration,
            cache_hits_added=successful
        )
    
    def run_all_strategies(self) -> Dict[str, WarmingResult]:
        """
        Run all enabled warming strategies
        
        Returns:
            Dictionary mapping strategy to result
        """
        results = {}
        
        # Popular symbols
        if self._should_warm_strategy('popular'):
            results['popular'] = self.warm_popular_symbols()
        
        # Time-based
        if self._should_warm_strategy('time_based'):
            results['time_based'] = self.warm_by_time_pattern()
        
        # Trending
        if self._should_warm_strategy('sector_rotation'):
            results['trending'] = self.warm_trending()
        
        return results
    
    def get_stats(self) -> Dict:
        """Get warming statistics"""
        return {
            'total_warmed': self.warming_stats['total_warmed'],
            'total_failed': self.warming_stats['total_failed'],
            'by_strategy': self.warming_stats['by_strategy'],
            'currently_warming': len(self.warming_queue),
            'last_warming': {
                strategy: datetime.fromtimestamp(ts).isoformat()
                for strategy, ts in self.last_warming.items()
            }
        }


# Global warmer instance
_global_warmer = None


def get_warmer() -> SmartCacheWarmer:
    """Get global cache warmer instance"""
    global _global_warmer
    if _global_warmer is None:
        _global_warmer = SmartCacheWarmer()
    return _global_warmer


if __name__ == "__main__":
    # Example usage
    import json
    
    warmer = SmartCacheWarmer()
    
    print("Running cache warming strategies...")
    print("=" * 60)
    
    # Warm popular symbols
    print("\n1. Warming popular symbols...")
    result = warmer.warm_popular_symbols(limit=10)
    print(f"   Warmed: {result.symbols_warmed}")
    print(f"   Failed: {result.symbols_failed}")
    print(f"   Duration: {result.duration_seconds:.1f}s")
    
    # Warm by time pattern
    print("\n2. Warming by time pattern...")
    result = warmer.warm_by_time_pattern(look_ahead_hours=1, limit=5)
    print(f"   Warmed: {result.symbols_warmed}")
    print(f"   Duration: {result.duration_seconds:.1f}s")
    
    # Get stats
    print("\n3. Warming Statistics:")
    print(json.dumps(warmer.get_stats(), indent=2))
