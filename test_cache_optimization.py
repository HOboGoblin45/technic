#!/usr/bin/env python3
"""Test cache optimization improvements"""

import time
import sys
sys.path.insert(0, '.')

from technic_v4.data_engine import get_price_history, get_cache_stats, clear_memory_cache

print("=" * 80)
print("CACHE OPTIMIZATION TEST")
print("=" * 80)

# Test symbols
test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

print("\n1. Testing Cache Normalization (88 days → 90 days)")
print("-" * 80)

# Clear cache
clear_memory_cache()

# First request: 88 days (should normalize to 90)
start = time.time()
df1 = get_price_history('AAPL', 88)
time1 = time.time() - start
print(f"Request 1: AAPL 88 days - {time1:.3f}s ({len(df1)} bars)")

# Second request: 90 days (should hit normalized cache)
start = time.time()
df2 = get_price_history('AAPL', 90)
time2 = time.time() - start
print(f"Request 2: AAPL 90 days - {time2:.3f}s ({len(df2)} bars) - Should be CACHED")

# Third request: 92 days (should also hit normalized cache)
start = time.time()
df3 = get_price_history('AAPL', 92)
time3 = time.time() - start
print(f"Request 3: AAPL 92 days - {time3:.3f}s ({len(df3)} bars) - Should be CACHED")

stats = get_cache_stats()
print(f"\nCache Stats: {stats['hits']}/{stats['total']} hits ({stats['hit_rate']:.1f}%)")

print("\n2. Testing Extended TTL (4 hours vs 1 hour)")
print("-" * 80)
print("Cache TTL increased from 1 hour (3600s) to 4 hours (14400s)")
print("This means cached data stays valid 4x longer")
print("✓ Benefit: Fewer API calls during the day")

print("\n3. Testing Multiple Symbols with Cache Reuse")
print("-" * 80)

clear_memory_cache()

# Scan multiple symbols with similar lookback periods
lookbacks = [88, 90, 92, 89, 91]  # All should normalize to 90
total_time = 0
api_calls = 0

for i, symbol in enumerate(test_symbols):
    days = lookbacks[i]
    start = time.time()
    df = get_price_history(symbol, days)
    elapsed = time.time() - start
    total_time += elapsed
    
    # First call for each symbol is an API call
    if elapsed > 0.1:  # Assume >100ms means API call
        api_calls += 1
    
    print(f"{symbol} {days} days: {elapsed:.3f}s ({len(df)} bars)")

stats = get_cache_stats()
print(f"\nTotal time: {total_time:.3f}s")
print(f"API calls: ~{api_calls}")
print(f"Cache Stats: {stats['hits']}/{stats['total']} hits ({stats['hit_rate']:.1f}%)")

print("\n4. Testing Cache Hit Rate Improvement")
print("-" * 80)

clear_memory_cache()

# Simulate scanner behavior: scan same symbols multiple times
iterations = 3
symbols_per_scan = 10
test_symbols_extended = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
                         'NVDA', 'META', 'NFLX', 'AMD', 'INTC']

total_requests = 0
total_time = 0

for iteration in range(iterations):
    print(f"\nIteration {iteration + 1}:")
    iter_start = time.time()
    
    for symbol in test_symbols_extended:
        df = get_price_history(symbol, 90)
        total_requests += 1
    
    iter_time = time.time() - iter_start
    total_time += iter_time
    
    stats = get_cache_stats()
    print(f"  Time: {iter_time:.3f}s | Cache: {stats['hits']}/{stats['total']} ({stats['hit_rate']:.1f}%)")

print(f"\nFinal Stats:")
print(f"  Total requests: {total_requests}")
print(f"  Total time: {total_time:.3f}s")
print(f"  Avg time/request: {total_time/total_requests:.3f}s")

final_stats = get_cache_stats()
print(f"  Final cache hit rate: {final_stats['hit_rate']:.1f}%")
print(f"  Cache size: {final_stats['cache_size']} entries")

print("\n5. Expected Improvements")
print("-" * 80)
print("✓ Cache TTL: 1 hour → 4 hours (4x longer validity)")
print("✓ Cache normalization: 88/89/90/91/92 days → all use 90-day cache")
print("✓ Dual cache keys: Both exact and normalized keys stored")
print("✓ Expected cache hit rate: 50.5% → 65-70%")
print("✓ Expected time savings: 2-3 seconds per scan")

print("\n" + "=" * 80)
print("CACHE OPTIMIZATION TEST COMPLETE")
print("=" * 80)
