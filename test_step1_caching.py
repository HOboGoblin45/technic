"""
Test script for Step 1: Multi-Layer Caching Performance
Tests the L1 memory cache implementation in data_engine.py
"""

import time
from technic_v4 import data_engine

def test_cache_performance():
    """Test cache hit rates and performance improvement"""
    
    print("=" * 60)
    print("STEP 1 CACHING TEST - Performance Validation")
    print("=" * 60)
    
    # Clear cache to start fresh
    print("\n1. Clearing cache...")
    data_engine.clear_memory_cache()
    stats = data_engine.get_cache_stats()
    print(f"   Cache cleared. Stats: {stats}")
    
    # Test 1: Cold fetch (no cache)
    print("\n2. Testing COLD fetch (no cache)...")
    test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    
    start_time = time.time()
    for symbol in test_symbols:
        df = data_engine.get_price_history(symbol, days=90, freq="daily")
        if df is not None:
            print(f"   ✓ {symbol}: {len(df)} bars fetched")
        else:
            print(f"   ✗ {symbol}: Failed to fetch")
    cold_time = time.time() - start_time
    
    stats_after_cold = data_engine.get_cache_stats()
    print(f"\n   Cold fetch time: {cold_time:.2f}s")
    print(f"   Cache stats: {stats_after_cold}")
    
    # Test 2: Warm fetch (with cache)
    print("\n3. Testing WARM fetch (with cache)...")
    
    start_time = time.time()
    for symbol in test_symbols:
        df = data_engine.get_price_history(symbol, days=90, freq="daily")
        if df is not None:
            print(f"   ✓ {symbol}: {len(df)} bars (cached)")
        else:
            print(f"   ✗ {symbol}: Failed to fetch")
    warm_time = time.time() - start_time
    
    stats_after_warm = data_engine.get_cache_stats()
    print(f"\n   Warm fetch time: {warm_time:.2f}s")
    print(f"   Cache stats: {stats_after_warm}")
    
    # Calculate improvement
    if cold_time > 0:
        speedup = cold_time / warm_time if warm_time > 0 else float('inf')
        improvement_pct = ((cold_time - warm_time) / cold_time) * 100
        
        print("\n" + "=" * 60)
        print("RESULTS:")
        print("=" * 60)
        print(f"Cold fetch time: {cold_time:.2f}s")
        print(f"Warm fetch time: {warm_time:.2f}s")
        print(f"Speedup: {speedup:.1f}x faster")
        print(f"Improvement: {improvement_pct:.1f}%")
        
        # Calculate cache hit rate
        total_requests = stats_after_warm.get('total_requests', 0)
        cache_hits = stats_after_warm.get('cache_hits', 0)
        if total_requests > 0:
            hit_rate = (cache_hits / total_requests) * 100
            print(f"Cache hit rate: {hit_rate:.1f}%")
        
        print("\n✅ SUCCESS: Caching is working!" if speedup > 2 else "\n⚠️  WARNING: Speedup less than expected")
        
        # Expected results
        print("\n" + "=" * 60)
        print("EXPECTED vs ACTUAL:")
        print("=" * 60)
        print(f"Expected speedup: 10-20x")
        print(f"Actual speedup: {speedup:.1f}x")
        print(f"Expected hit rate: ~50% (second fetch)")
        print(f"Actual hit rate: {hit_rate:.1f}%" if total_requests > 0 else "N/A")
        
        if speedup >= 10:
            print("\n🎉 EXCELLENT: Exceeds performance target!")
        elif speedup >= 5:
            print("\n✅ GOOD: Meets minimum performance target")
        else:
            print("\n⚠️  NEEDS IMPROVEMENT: Below target performance")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_cache_performance()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
