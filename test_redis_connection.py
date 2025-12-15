#!/usr/bin/env python3
"""
Test Redis Connection for Technic Scanner
==========================================

This script tests the Redis connection and verifies it's working correctly.

Usage:
    python test_redis_connection.py

Expected output:
    ✅ Redis connection successful!
    ✅ Set/Get test: test_value
    ✅ Redis version: 7.x.x
    ✅ Memory used: X MB
"""

import os
import sys


def test_redis_connection():
    """Test Redis connection and basic operations"""
    
    print("=" * 80)
    print("REDIS CONNECTION TEST")
    print("=" * 80)
    print()
    
    # Check environment variables
    print("1. Checking environment variables...")
    redis_url = os.getenv('REDIS_URL')
    redis_host = os.getenv('REDIS_HOST')
    redis_port = os.getenv('REDIS_PORT')
    
    if not redis_url:
        print("❌ REDIS_URL not set")
        print("   Please set REDIS_URL environment variable")
        print("   Example: redis://:password@host:port/0")
        return False
    
    print(f"✅ REDIS_URL: {redis_url[:20]}...{redis_url[-20:]}")  # Hide password
    if redis_host:
        print(f"✅ REDIS_HOST: {redis_host}")
    if redis_port:
        print(f"✅ REDIS_PORT: {redis_port}")
    print()
    
    # Try to import redis
    print("2. Checking redis package...")
    try:
        import redis
        print(f"✅ redis package installed (version {redis.__version__})")
    except ImportError:
        print("❌ redis package not installed")
        print("   Run: pip install redis hiredis")
        return False
    print()
    
    # Test connection
    print("3. Testing Redis connection...")
    try:
        client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=10,
            socket_timeout=10
        )
        
        # Ping test
        client.ping()
        print("✅ Redis connection successful!")
        print()
        
        # Test set/get
        print("4. Testing set/get operations...")
        test_key = 'technic_test_key'
        test_value = 'technic_test_value'
        
        client.set(test_key, test_value, ex=60)
        retrieved = client.get(test_key)
        
        if retrieved == test_value:
            print(f"✅ Set/Get test passed: '{retrieved}'")
        else:
            print(f"❌ Set/Get test failed: expected '{test_value}', got '{retrieved}'")
            return False
        
        # Clean up test key
        client.delete(test_key)
        print()
        
        # Get server info
        print("5. Redis server information...")
        info = client.info('server')
        print(f"✅ Redis version: {info.get('redis_version', 'unknown')}")
        print(f"✅ OS: {info.get('os', 'unknown')}")
        print(f"✅ Uptime: {info.get('uptime_in_days', 0)} days")
        print()
        
        # Get memory info
        print("6. Redis memory information...")
        mem_info = client.info('memory')
        print(f"✅ Memory used: {mem_info.get('used_memory_human', 'unknown')}")
        print(f"✅ Memory peak: {mem_info.get('used_memory_peak_human', 'unknown')}")
        print(f"✅ Memory limit: {mem_info.get('maxmemory_human', 'unlimited')}")
        print()
        
        # Get stats
        print("7. Redis statistics...")
        stats_info = client.info('stats')
        print(f"✅ Total connections: {stats_info.get('total_connections_received', 0)}")
        print(f"✅ Total commands: {stats_info.get('total_commands_processed', 0)}")
        print(f"✅ Keyspace hits: {stats_info.get('keyspace_hits', 0)}")
        print(f"✅ Keyspace misses: {stats_info.get('keyspace_misses', 0)}")
        
        hits = stats_info.get('keyspace_hits', 0)
        misses = stats_info.get('keyspace_misses', 0)
        total = hits + misses
        if total > 0:
            hit_rate = (hits / total) * 100
            print(f"✅ Cache hit rate: {hit_rate:.1f}%")
        print()
        
        # Get database info
        print("8. Database information...")
        db_size = client.dbsize()
        print(f"✅ Total keys in database: {db_size}")
        print()
        
        print("=" * 80)
        print("✅ ALL TESTS PASSED - Redis is ready to use!")
        print("=" * 80)
        print()
        print("Next steps:")
        print("1. Deploy your scanner with Redis integration")
        print("2. Run first scan (will populate Redis cache)")
        print("3. Run second scan (should be 3-4x faster!)")
        print()
        
        return True
        
    except redis.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        print()
        print("Troubleshooting:")
        print("- Check REDIS_URL is correct")
        print("- Verify Redis instance is running")
        print("- Check network connectivity")
        print("- Verify password is correct")
        return False
        
    except redis.AuthenticationError as e:
        print(f"❌ Authentication error: {e}")
        print()
        print("Troubleshooting:")
        print("- Check REDIS_PASSWORD is correct")
        print("- Verify password in REDIS_URL matches Redis dashboard")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print()
        print("Troubleshooting:")
        print("- Check all environment variables are set")
        print("- Verify redis package is installed")
        print("- Check Render logs for more details")
        return False


def test_technic_cache():
    """Test Technic's Redis cache wrapper"""
    
    print("=" * 80)
    print("TECHNIC REDIS CACHE TEST")
    print("=" * 80)
    print()
    
    try:
        from technic_v4.cache.redis_cache import get_redis_cache
        
        print("1. Initializing Technic Redis cache...")
        cache = get_redis_cache()
        
        if not cache.enabled:
            print("❌ Redis cache not enabled")
            print("   Check REDIS_URL environment variable")
            return False
        
        print("✅ Technic Redis cache initialized")
        print()
        
        # Test price data caching
        print("2. Testing price data cache...")
        import pandas as pd
        
        # Create test data
        test_df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=10),
            'Close': [100 + i for i in range(10)],
            'Volume': [1000000 + i*10000 for i in range(10)]
        })
        
        # Cache it
        cache.set_price_data('TEST_SYMBOL', 150, test_df, ttl_hours=1)
        print("✅ Cached test price data")
        
        # Retrieve it
        retrieved_df = cache.get_price_data('TEST_SYMBOL', 150)
        
        if retrieved_df is not None and len(retrieved_df) == 10:
            print("✅ Retrieved test price data successfully")
        else:
            print("❌ Failed to retrieve test price data")
            return False
        print()
        
        # Get stats
        print("3. Cache statistics...")
        stats = cache.get_stats()
        
        if stats.get('enabled'):
            print(f"✅ Total keys: {stats.get('total_keys', 0)}")
            print(f"✅ Cache hits: {stats.get('hits', 0)}")
            print(f"✅ Cache misses: {stats.get('misses', 0)}")
            print(f"✅ Hit rate: {stats.get('hit_rate', 0):.1f}%")
            print(f"✅ Memory used: {stats.get('memory_used', 'unknown')}")
        else:
            print("❌ Could not get cache stats")
            return False
        print()
        
        # Health check
        print("4. Health check...")
        if cache.health_check():
            print("✅ Redis is healthy and responsive")
        else:
            print("❌ Redis health check failed")
            return False
        print()
        
        print("=" * 80)
        print("✅ ALL TECHNIC CACHE TESTS PASSED!")
        print("=" * 80)
        print()
        print("Redis is ready for production use!")
        print()
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running from the project root")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    print()
    
    # Test 1: Basic Redis connection
    success1 = test_redis_connection()
    
    print()
    print()
    
    # Test 2: Technic cache wrapper
    success2 = test_technic_cache()
    
    print()
    
    if success1 and success2:
        print("🎉 SUCCESS! Redis is fully configured and ready!")
        print()
        print("Next steps:")
        print("1. Update data_engine.py to use Redis cache")
        print("2. Deploy to Render")
        print("3. Run first scan (populate cache)")
        print("4. Run second scan (see 3-4x speedup!)")
        sys.exit(0)
    else:
        print("❌ FAILED! Please fix the issues above")
        sys.exit(1)
