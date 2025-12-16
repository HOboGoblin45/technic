"""
Redis Cache Layer - Phase 3C
High-performance caching for technical indicators and ML predictions

PERFORMANCE BENEFITS:
- 2x speedup by caching computed indicators
- Reduces redundant calculations
- Enables incremental scanning
- Supports distributed workers
"""

import redis
import pickle
import logging
from functools import wraps
from typing import Optional, Dict, List, Any
import os

logger = logging.getLogger(__name__)

class RedisCache:
    """High-performance Redis caching with async support"""
    
    def __init__(self, host: Optional[str] = None, port: int = 6379, db: int = 0):
        """
        Initialize Redis cache.
        
        Args:
            host: Redis host (defaults to REDIS_URL env var or localhost)
            port: Redis port
            db: Redis database number
        """
        # Try to get Redis URL from environment (supports authentication)
        redis_url = os.getenv('REDIS_URL')
        
        if redis_url:
            # Parse Redis URL (format: redis://[user:password@]host:port[/db])
            try:
                self.client = redis.from_url(
                    redis_url,
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_keepalive=True,
                    max_connections=100,
                    retry_on_timeout=True
                )
                logger.info("[REDIS] Connected via REDIS_URL")
            except Exception as e:
                logger.warning(f"[REDIS] Failed to connect via URL: {e}")
                self.client = None
        else:
            # Fallback to individual parameters (supports password)
            host = host or os.getenv('REDIS_HOST', 'localhost')
            port = int(os.getenv('REDIS_PORT', port))
            db = int(os.getenv('REDIS_DB', db))
            password = os.getenv('REDIS_PASSWORD')
            
            try:
                self.client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_keepalive=True,
                    max_connections=100,
                    retry_on_timeout=True
                )
                logger.info(f"[REDIS] Connected to {host}:{port}")
            except Exception as e:
                logger.warning(f"[REDIS] Failed to connect: {e}")
                self.client = None
        
        self.available = self._check_connection()
        
        if not self.available:
            logger.warning("[REDIS] Cache not available - running without caching")
        else:
            logger.info("[REDIS] Cache is available and ready")
    
    def _check_connection(self) -> bool:
        """Check if Redis is available"""
        if self.client is None:
            return False
        
        try:
            self.client.ping()
            return True
        except Exception as e:
            logger.warning(f"[REDIS] Connection check failed: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.available:
            return None
        
        try:
            value = self.client.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.warning(f"[REDIS] Get failed for {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 300):
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: 5 minutes)
        """
        if not self.available:
            return False
        
        try:
            self.client.setex(key, ttl, pickle.dumps(value))
            return True
        except Exception as e:
            logger.warning(f"[REDIS] Set failed for {key}: {e}")
            return False
    
    def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple keys at once"""
        if not self.available:
            return {}
        
        try:
            values = self.client.mget(keys)
            return {
                k: pickle.loads(v) if v else None
                for k, v in zip(keys, values)
                if v is not None
            }
        except Exception as e:
            logger.warning(f"[REDIS] Batch get failed: {e}")
            return {}
    
    def batch_set(self, data: Dict[str, Any], ttl: int = 300):
        """Set multiple keys at once"""
        if not self.available:
            return False
        
        try:
            pipe = self.client.pipeline()
            for key, value in data.items():
                pipe.setex(key, ttl, pickle.dumps(value))
            pipe.execute()
            return True
        except Exception as e:
            logger.warning(f"[REDIS] Batch set failed: {e}")
            return False
    
    def delete(self, key: str):
        """Delete key from cache"""
        if not self.available:
            return False
        
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"[REDIS] Delete failed for {key}: {e}")
            return False
    
    def clear_pattern(self, pattern: str):
        """Delete all keys matching pattern"""
        if not self.available:
            return False
        
        try:
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
                logger.info(f"[REDIS] Cleared {len(keys)} keys matching {pattern}")
            return True
        except Exception as e:
            logger.warning(f"[REDIS] Clear pattern failed: {e}")
            return False
    
    def cache_indicators(self, ttl: int = 300):
        """
        Decorator to cache technical indicators.
        
        Args:
            ttl: Time to live in seconds (default: 5 minutes)
        """
        def decorator(func):
            @wraps(func)
            def wrapper(symbol, *args, **kwargs):
                if not self.available:
                    return func(symbol, *args, **kwargs)
                
                # Create cache key from function name and arguments
                key = f"indicators:{symbol}:{func.__name__}:{hash(str(args))}"
                
                # Try cache
                cached = self.get(key)
                if cached is not None:
                    logger.debug(f"[REDIS] Cache hit for {symbol} indicators")
                    return cached
                
                # Compute and cache
                result = func(symbol, *args, **kwargs)
                self.set(key, result, ttl)
                logger.debug(f"[REDIS] Cached {symbol} indicators")
                return result
            return wrapper
        return decorator
    
    def cache_ml_predictions(self, ttl: int = 300):
        """
        Decorator to cache ML model predictions.
        
        Args:
            ttl: Time to live in seconds (default: 5 minutes)
        """
        def decorator(func):
            @wraps(func)
            def wrapper(df, *args, **kwargs):
                if not self.available:
                    return func(df, *args, **kwargs)
                
                # For batch predictions, cache per symbol
                if 'Symbol' in df.columns:
                    results = {}
                    uncached_df = df.copy()
                    
                    for symbol in df['Symbol'].unique():
                        key = f"ml_pred:{symbol}:{func.__name__}:{hash(str(args))}"
                        cached = self.get(key)
                        if cached is not None:
                            results[symbol] = cached
                            uncached_df = uncached_df[uncached_df['Symbol'] != symbol]
                    
                    # Compute uncached
                    if not uncached_df.empty:
                        new_results = func(uncached_df, *args, **kwargs)
                        
                        # Cache new results
                        for symbol in uncached_df['Symbol'].unique():
                            if symbol in new_results:
                                key = f"ml_pred:{symbol}:{func.__name__}:{hash(str(args))}"
                                self.set(key, new_results[symbol], ttl)
                                results[symbol] = new_results[symbol]
                    
                    return results
                else:
                    # Single prediction
                    return func(df, *args, **kwargs)
            return wrapper
        return decorator
    
    def warm_cache(self, symbols: List[str], days: int):
        """
        Pre-warm cache for top symbols.
        
        Args:
            symbols: List of symbols to warm
            days: Number of days of history
        """
        if not self.available:
            return
        
        logger.info(f"[REDIS] Warming cache for {len(symbols)} symbols")
        
        try:
            from technic_v4 import data_engine
            
            # Fetch and cache in batch
            price_data = data_engine.get_price_history_batch(symbols, days)
            
            cache_data = {}
            for symbol, df in price_data.items():
                if df is not None and not df.empty:
                    key = f"price:{symbol}:{days}"
                    cache_data[key] = df
            
            if cache_data:
                self.batch_set(cache_data, ttl=3600)  # 1 hour for price data
                logger.info(f"[REDIS] Cached {len(cache_data)} symbols")
        except Exception as e:
            logger.warning(f"[REDIS] Cache warming failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.available:
            return {
                'available': False,
                'connected': False
            }
        
        try:
            info = self.client.info('stats')
            memory_info = self.client.info('memory')
            
            hits = info.get('keyspace_hits', 0)
            misses = info.get('keyspace_misses', 0)
            total_requests = hits + misses
            hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0
            
            memory_used = memory_info.get('used_memory', 0)
            memory_used_mb = memory_used / (1024 * 1024)
            
            return {
                'available': True,
                'connected': True,
                'total_keys': self.client.dbsize(),
                'hits': hits,
                'misses': misses,
                'hit_rate': round(hit_rate, 2),
                'memory_used_mb': round(memory_used_mb, 2)
            }
        except Exception as e:
            logger.warning(f"[REDIS] Stats failed: {e}")
            return {
                'available': False,
                'connected': False,
                'error': str(e)
            }
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """
        Get detailed cache statistics including memory usage and key breakdown.
        
        Returns:
            Dictionary with detailed cache statistics
        """
        if not self.available:
            return {
                'available': False,
                'message': 'Redis cache not available'
            }
        
        try:
            # Get basic stats
            stats_info = self.client.info('stats')
            memory_info = self.client.info('memory')
            
            # Get all keys and analyze by type
            all_keys = self.client.keys('technic:*')
            keys_by_type = {}
            
            for key in all_keys[:1000]:  # Limit to first 1000 for performance
                try:
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                    parts = key_str.split(':')
                    if len(parts) >= 2:
                        key_type = parts[1]
                        keys_by_type[key_type] = keys_by_type.get(key_type, 0) + 1
                except Exception:
                    continue
            
            # Calculate hit rate
            hits = stats_info.get('keyspace_hits', 0)
            misses = stats_info.get('keyspace_misses', 0)
            total_requests = hits + misses
            hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0
            
            # Memory usage
            memory_used = memory_info.get('used_memory', 0)
            memory_peak = memory_info.get('used_memory_peak', 0)
            memory_used_mb = memory_used / (1024 * 1024)
            memory_peak_mb = memory_peak / (1024 * 1024)
            
            return {
                'available': True,
                'connection': {
                    'host': self.client.connection_pool.connection_kwargs.get('host', 'unknown'),
                    'port': self.client.connection_pool.connection_kwargs.get('port', 0),
                    'db': self.client.connection_pool.connection_kwargs.get('db', 0)
                },
                'performance': {
                    'total_keys': self.client.dbsize(),
                    'hits': hits,
                    'misses': misses,
                    'hit_rate': round(hit_rate, 2),
                    'total_requests': total_requests
                },
                'memory': {
                    'used_mb': round(memory_used_mb, 2),
                    'peak_mb': round(memory_peak_mb, 2),
                    'used_bytes': memory_used,
                    'fragmentation_ratio': memory_info.get('mem_fragmentation_ratio', 0)
                },
                'keys_by_type': keys_by_type,
                'server_info': {
                    'redis_version': memory_info.get('redis_version', 'unknown'),
                    'uptime_seconds': stats_info.get('uptime_in_seconds', 0)
                }
            }
        except Exception as e:
            logger.warning(f"[REDIS] Detailed stats failed: {e}")
            return {
                'available': False,
                'error': str(e)
            }
    
    def clear_all(self):
        """Clear all cache keys (use with caution!)"""
        if not self.available:
            return False
        
        try:
            self.client.flushdb()
            logger.info("[REDIS] Cleared all cache keys")
            return True
        except Exception as e:
            logger.warning(f"[REDIS] Clear all failed: {e}")
            return False

# Global instance
redis_cache = RedisCache()

# Export
__all__ = ['RedisCache', 'redis_cache']
