"""
Technic Cache Module
====================

Multi-layer caching for optimal performance:
- L1: In-memory dict (instant)
- L2: LRU cache (very fast)
- L3: Redis (persistent, shared)
"""

from .redis_cache import get_redis_cache, get_cache_stats, clear_all_cache

__all__ = ['get_redis_cache', 'get_cache_stats', 'clear_all_cache']
