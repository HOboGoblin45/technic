"""
Cache module for Technic Scanner
Provides Redis-based caching for performance optimization
"""

from .redis_cache import RedisCache, redis_cache

__all__ = ['RedisCache', 'redis_cache']
