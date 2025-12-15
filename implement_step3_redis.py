"""
Script to implement Step 3: Redis Distributed Caching
Adds L3 Redis cache layer to data_engine.py
"""

def implement_step3():
    """Add Redis caching to data_engine.py"""
    
    print("Implementing Step 3: Redis Distributed Caching...")
    
    # Read the current file
    with open('technic_v4/data_engine.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already implemented
    if 'import redis' in content:
        print("✓ Redis caching already implemented!")
        return
    
    # Add Redis import after other imports
    import_marker = "from technic_v4.infra.logging import get_logger"
    if import_marker in content:
        redis_imports = """from technic_v4.infra.logging import get_logger
import redis
from redis.exceptions import RedisError"""
        content = content.replace(import_marker, redis_imports)
        print("  ✓ Added Redis imports")
    
    # Add Redis client initialization after logger
    logger_marker = "logger = get_logger()"
    if logger_marker in content:
        redis_init = """logger = get_logger()

# L3 Redis cache (distributed, cross-process)
_REDIS_CLIENT = None
_REDIS_ENABLED = False

def _init_redis():
    \"\"\"Initialize Redis client (best-effort)\"\"\"
    global _REDIS_CLIENT, _REDIS_ENABLED
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
    \"\"\"Get Redis client (lazy init)\"\"\"
    global _REDIS_CLIENT, _REDIS_ENABLED
    if _REDIS_CLIENT is None:
        _init_redis()
    return _REDIS_CLIENT if _REDIS_ENABLED else None"""
        
        content = content.replace(logger_marker, redis_init)
        print("  ✓ Added Redis initialization")
    
    # Find the get_price_history function and add Redis caching
    # We'll add Redis check before L2 cache
    l2_check_marker = "    # L2: MarketCache (persistent disk cache)"
    if l2_check_marker in content:
        redis_check = """    # L3: Redis cache (distributed, cross-process)
    redis_client = get_redis_client()
    if redis_client:
        try:
            redis_key = f"price:{symbol}:{days}:{freq}"
            cached_data = redis_client.get(redis_key)
            if cached_data:
                import pickle
                df = pickle.loads(cached_data)
                logger.info("[data_engine] L3 Redis cache hit for %s (%d bars)", symbol, len(df))
                # Store in L1 for even faster access next time
                _MEMORY_CACHE[cache_key] = df
                _CACHE_STATS['hits'] += 1
                _CACHE_STATS['total'] += 1
                return df
        except RedisError as e:
            logger.warning("[REDIS] Cache read failed: %s", e)
        except Exception as e:
            logger.warning("[REDIS] Unexpected error: %s", e)
    
    # L2: MarketCache (persistent disk cache)"""
        
        content = content.replace(l2_check_marker, redis_check)
        print("  ✓ Added Redis cache read logic")
    
    # Add Redis cache write after successful fetch
    # Find where we return the dataframe after fetching
    write_marker = "    # Store in L1 memory cache\n    _MEMORY_CACHE[cache_key] = df"
    if write_marker in content:
        redis_write = """    # Store in L1 memory cache
    _MEMORY_CACHE[cache_key] = df
    
    # Store in L3 Redis cache (with 1 hour TTL)
    redis_client = get_redis_client()
    if redis_client and df is not None:
        try:
            import pickle
            redis_key = f"price:{symbol}:{days}:{freq}"
            redis_client.setex(redis_key, 3600, pickle.dumps(df))  # 1 hour TTL
            logger.debug("[REDIS] Cached %s in Redis", symbol)
        except RedisError as e:
            logger.warning("[REDIS] Cache write failed: %s", e)
        except Exception as e:
            logger.warning("[REDIS] Unexpected error: %s", e)"""
        
        content = content.replace(write_marker, redis_write)
        print("  ✓ Added Redis cache write logic")
    
    # Add Redis stats to get_cache_stats
    stats_marker = "def get_cache_stats() -> dict:"
    if stats_marker in content:
        # Find the return statement in get_cache_stats
        stats_return = """    return {
        'hits': _CACHE_STATS['hits'],
        'misses': _CACHE_STATS['misses'],
        'total': _CACHE_STATS['total'],
        'hit_rate': hit_rate,
        'cache_size': len(_MEMORY_CACHE),
    }"""
        
        new_stats_return = """    # Redis stats (best-effort)
    redis_info = {}
    redis_client = get_redis_client()
    if redis_client:
        try:
            info = redis_client.info('stats')
            redis_info = {
                'redis_enabled': True,
                'redis_keys': redis_client.dbsize(),
                'redis_hits': info.get('keyspace_hits', 0),
                'redis_misses': info.get('keyspace_misses', 0),
            }
        except Exception:
            redis_info = {'redis_enabled': False}
    
    return {
        'hits': _CACHE_STATS['hits'],
        'misses': _CACHE_STATS['misses'],
        'total': _CACHE_STATS['total'],
        'hit_rate': hit_rate,
        'cache_size': len(_MEMORY_CACHE),
        **redis_info,
    }"""
        
        content = content.replace(stats_return, new_stats_return)
        print("  ✓ Added Redis stats to get_cache_stats()")
    
    # Add clear_redis_cache function
    clear_marker = "def clear_memory_cache():"
    if clear_marker in content:
        redis_clear = """def clear_redis_cache():
    \"\"\"Clear Redis cache (best-effort)\"\"\"
    redis_client = get_redis_client()
    if redis_client:
        try:
            redis_client.flushdb()
            logger.info("[REDIS] Redis cache cleared")
        except Exception as e:
            logger.warning("[REDIS] Failed to clear cache: %s", e)

def clear_memory_cache():"""
        
        content = content.replace(clear_marker, redis_clear)
        print("  ✓ Added clear_redis_cache() function")
    
    # Write the updated content
    with open('technic_v4/data_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✅ Step 3 implementation complete!")
    print("\nNext steps:")
    print("  1. Install Redis: pip install redis")
    print("  2. Start Redis server: redis-server")
    print("  3. Test Redis caching")

if __name__ == "__main__":
    try:
        implement_step3()
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
