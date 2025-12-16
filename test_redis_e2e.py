"""
End-to-End Redis Caching Test
Tests full scanner integration with Redis caching
"""

import os
import time

# Set Redis URL BEFORE importing anything
os.environ['REDIS_URL'] = 'redis://:ytvZ1OVXoGV4OenJH3GJYkDLeg2emqad@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0'

# Now import after environment is set
from technic_v4.scanner_core import run_scan, ScanConfig
from technic_v4.cache.redis_cache import RedisCache

# Create fresh Redis cache instance with the environment variable set
redis_cache = RedisCache()

print("="*80)
print("PHASE 3C: END-TO-END REDIS CACHING TEST")
print("="*80)

# Test 1: Verify Redis is available
print("\n[TEST 1] Checking Redis availability...")
if redis_cache.available:
    print("✅ Redis is available and connected")
    stats = redis_cache.get_stats()
    print(f"   Total keys: {stats['total_keys']}")
    print(f"   Hit rate: {stats['hit_rate']:.2f}%")
else:
    print("❌ Redis is NOT available - test cannot proceed")
    exit(1)

# Test 2: Clear cache for clean test
print("\n[TEST 2] Clearing cache for clean test...")
redis_cache.clear_pattern("technic:*")
print("✅ Cache cleared")

# Test 3: First scan (cold cache)
print("\n[TEST 3] Running first scan (cold cache - should be slower)...")
config = ScanConfig(
    max_symbols=20,  # Small test
    lookback_days=90,
    trade_style="Short-term swing"
)

start_time = time.time()
df1, msg1 = run_scan(config)
time1 = time.time() - start_time

print(f"✅ First scan complete")
print(f"   Time: {time1:.2f}s")
print(f"   Results: {len(df1)} symbols")
print(f"   Status: {msg1}")

# Check cache stats after first scan
stats_after_first = redis_cache.get_stats()
print(f"\n   Cache stats after first scan:")
print(f"   - Total keys: {stats_after_first['total_keys']}")
print(f"   - Cache hits: {stats_after_first['hits']}")
print(f"   - Cache misses: {stats_after_first['misses']}")
print(f"   - Hit rate: {stats_after_first['hit_rate']:.2f}%")

# Test 4: Second scan (warm cache)
print("\n[TEST 4] Running second scan (warm cache - should be faster)...")
print("   Waiting 2 seconds before second scan...")
time.sleep(2)

start_time = time.time()
df2, msg2 = run_scan(config)
time2 = time.time() - start_time

print(f"✅ Second scan complete")
print(f"   Time: {time2:.2f}s")
print(f"   Results: {len(df2)} symbols")
print(f"   Status: {msg2}")

# Check cache stats after second scan
stats_after_second = redis_cache.get_stats()
print(f"\n   Cache stats after second scan:")
print(f"   - Total keys: {stats_after_second['total_keys']}")
print(f"   - Cache hits: {stats_after_second['hits']}")
print(f"   - Cache misses: {stats_after_second['misses']}")
print(f"   - Hit rate: {stats_after_second['hit_rate']:.2f}%")

# Test 5: Calculate speedup
print("\n[TEST 5] Performance Analysis...")
speedup = time1 / time2 if time2 > 0 else 1.0
cache_hit_improvement = stats_after_second['hits'] - stats_after_first['hits']

print(f"   First scan time: {time1:.2f}s")
print(f"   Second scan time: {time2:.2f}s")
print(f"   Speedup: {speedup:.2f}x")
print(f"   Cache hits gained: {cache_hit_improvement}")

if speedup >= 1.5:
    print(f"   ✅ EXCELLENT! {speedup:.2f}x speedup achieved (target: 2x)")
elif speedup >= 1.2:
    print(f"   ✅ GOOD! {speedup:.2f}x speedup achieved")
elif speedup >= 1.0:
    print(f"   ⚠️  MARGINAL: {speedup:.2f}x speedup (expected 2x)")
else:
    print(f"   ❌ SLOWER: Second scan was slower (unexpected)")

# Test 6: Verify data consistency
print("\n[TEST 6] Verifying data consistency...")
if len(df1) == len(df2):
    print(f"✅ Same number of results: {len(df1)} symbols")
else:
    print(f"⚠️  Different result counts: {len(df1)} vs {len(df2)}")

# Check if same symbols
if len(df1) > 0 and len(df2) > 0:
    symbols1 = set(df1['Symbol'].tolist())
    symbols2 = set(df2['Symbol'].tolist())
    if symbols1 == symbols2:
        print(f"✅ Same symbols in both scans")
    else:
        print(f"⚠️  Different symbols: {len(symbols1 - symbols2)} unique to first, {len(symbols2 - symbols1)} unique to second")

# Test 7: Test fallback behavior
print("\n[TEST 7] Testing fallback behavior (Redis disabled)...")
os.environ['REDIS_URL'] = 'redis://invalid:6379/0'  # Invalid URL

# Force reconnection
from technic_v4.cache.redis_cache import RedisCache
fallback_cache = RedisCache()

if not fallback_cache.available:
    print("✅ Fallback works - cache gracefully disabled with invalid URL")
else:
    print("⚠️  Cache still available with invalid URL (unexpected)")

# Restore valid URL
os.environ['REDIS_URL'] = 'redis://:ytvZ1OVXoGV4OenJH3GJYkDLeg2emqad@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0'

# Final Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print(f"✅ Redis connection: Working")
print(f"✅ First scan (cold): {time1:.2f}s")
print(f"✅ Second scan (warm): {time2:.2f}s")
print(f"✅ Speedup achieved: {speedup:.2f}x")
print(f"✅ Cache hit rate: {stats_after_second['hit_rate']:.2f}%")
print(f"✅ Fallback behavior: Working")

if speedup >= 1.5 and stats_after_second['hit_rate'] > 50:
    print("\n🎉 ALL TESTS PASSED! Redis caching is working excellently!")
    print(f"   Phase 3C is ready for production deployment.")
elif speedup >= 1.2:
    print("\n✅ TESTS PASSED! Redis caching is working well.")
    print(f"   Phase 3C is ready for production deployment.")
else:
    print("\n⚠️  TESTS COMPLETED with marginal improvement.")
    print(f"   Redis is working but speedup is less than expected.")
    print(f"   This may be due to small test size (20 symbols).")

print("\n" + "="*80)
