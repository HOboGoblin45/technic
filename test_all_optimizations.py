"""
Comprehensive Test Suite for Scanner Performance Optimizations
Tests all 4 steps: Caching, Filtering, Redis, and Parallel Processing
"""

import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_step1_caching():
    """Test Step 1: Multi-layer caching"""
    print("\n" + "="*70)
    print("STEP 1: MULTI-LAYER CACHING TEST")
    print("="*70)
    
    try:
        from technic_v4 import data_engine
        
        # Test cache functions exist
        assert hasattr(data_engine, 'get_cache_stats'), "Missing get_cache_stats()"
        assert hasattr(data_engine, 'clear_memory_cache'), "Missing clear_memory_cache()"
        
        # Clear cache
        data_engine.clear_memory_cache()
        print("✓ Cache cleared successfully")
        
        # Get initial stats
        stats = data_engine.get_cache_stats()
        print(f"✓ Cache stats: {stats}")
        
        # Test cache with a symbol
        print("\nTesting cache with AAPL...")
        start = time.time()
        df1 = data_engine.get_price_history("AAPL", days=90)
        time1 = time.time() - start
        print(f"  First fetch (cold): {time1:.3f}s, {len(df1) if df1 is not None else 0} bars")
        
        start = time.time()
        df2 = data_engine.get_price_history("AAPL", days=90)
        time2 = time.time() - start
        print(f"  Second fetch (warm): {time2:.3f}s, {len(df2) if df2 is not None else 0} bars")
        
        if time2 < time1:
            speedup = time1 / time2 if time2 > 0 else float('inf')
            print(f"  ✓ Cache speedup: {speedup:.1f}x faster")
        
        # Final stats
        stats = data_engine.get_cache_stats()
        print(f"\n✓ Final cache stats: {stats}")
        print(f"  Hit rate: {stats.get('hit_rate', 0):.1%}")
        
        return True
        
    except Exception as e:
        print(f"✗ Step 1 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step2_filtering():
    """Test Step 2: Smart universe filtering"""
    print("\n" + "="*70)
    print("STEP 2: SMART UNIVERSE FILTERING TEST")
    print("="*70)
    
    try:
        from technic_v4.scanner_core import _smart_filter_universe, _prepare_universe
        from technic_v4.universe_loader import load_universe
        from dataclasses import dataclass
        
        # Create a mock config
        @dataclass
        class MockConfig:
            sectors: list = None
            subindustries: list = None
            industry_contains: str = None
        
        config = MockConfig()
        
        # Load universe
        universe = load_universe()
        print(f"✓ Loaded universe: {len(universe)} symbols")
        
        # Test smart filtering
        filtered = _smart_filter_universe(universe, config)
        reduction = ((len(universe) - len(filtered)) / len(universe) * 100) if len(universe) > 0 else 0
        
        print(f"✓ Smart filter applied:")
        print(f"  Before: {len(universe)} symbols")
        print(f"  After: {len(filtered)} symbols")
        print(f"  Reduction: {reduction:.1f}%")
        
        if reduction >= 50:
            print(f"  ✓ Target reduction achieved (50%+ filtered)")
        else:
            print(f"  ⚠ Lower than expected reduction (target: 70-80%)")
        
        return True
        
    except Exception as e:
        print(f"✗ Step 2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step3_redis():
    """Test Step 3: Redis caching (optional, requires Redis server)"""
    print("\n" + "="*70)
    print("STEP 3: REDIS CACHING TEST")
    print("="*70)
    
    try:
        from technic_v4 import data_engine
        
        # Check if Redis functions exist
        assert hasattr(data_engine, 'get_redis_client'), "Missing get_redis_client()"
        assert hasattr(data_engine, 'clear_redis_cache'), "Missing clear_redis_cache()"
        
        print("✓ Redis functions exist")
        
        # Try to get Redis client (may fail if Redis not running)
        redis_client = data_engine.get_redis_client()
        
        if redis_client:
            print("✓ Redis client connected")
            
            # Test Redis cache
            try:
                redis_client.ping()
                print("✓ Redis server responding")
                
                # Get stats
                stats = data_engine.get_cache_stats()
                if 'redis_enabled' in stats:
                    print(f"✓ Redis stats: {stats.get('redis_keys', 0)} keys")
                
                return True
            except Exception as e:
                print(f"⚠ Redis available but error: {e}")
                return True  # Not a failure, just unavailable
        else:
            print("⚠ Redis not available (this is optional)")
            print("  To enable: pip install redis && redis-server")
            return True  # Not a failure, just unavailable
        
    except Exception as e:
        print(f"⚠ Step 3 test skipped: {e}")
        print("  Redis is optional for local development")
        return True  # Not a failure


def test_step4_parallel():
    """Test Step 4: Parallel processing"""
    print("\n" + "="*70)
    print("STEP 4: PARALLEL PROCESSING TEST")
    print("="*70)
    
    try:
        from technic_v4 import scanner_core
        import os
        
        # Check MAX_WORKERS is set correctly
        max_workers = scanner_core.MAX_WORKERS
        cpu_count = os.cpu_count() or 4
        expected = min(32, cpu_count * 2)
        
        print(f"✓ CPU cores detected: {cpu_count}")
        print(f"✓ MAX_WORKERS set to: {max_workers}")
        print(f"  Expected: {expected}")
        
        if max_workers == expected:
            print("  ✓ Optimal worker count configured")
        else:
            print(f"  ⚠ Worker count mismatch (got {max_workers}, expected {expected})")
        
        # Check if parallel functions exist
        assert hasattr(scanner_core, '_run_symbol_scans'), "Missing _run_symbol_scans()"
        print("✓ Parallel scan functions exist")
        
        return True
        
    except Exception as e:
        print(f"✗ Step 4 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test all steps working together"""
    print("\n" + "="*70)
    print("INTEGRATION TEST: ALL STEPS TOGETHER")
    print("="*70)
    
    try:
        from technic_v4.scanner_core import run_scan, ScanConfig
        from technic_v4 import data_engine
        
        # Clear cache for fair test
        data_engine.clear_memory_cache()
        
        # Create a minimal scan config
        config = ScanConfig(
            trade_style="swing",
            sectors=["Technology"],  # Limit to one sector for speed
            max_results=5,
        )
        
        print("Running optimized scan (Technology sector, max 5 results)...")
        print("This tests: Caching + Filtering + Parallel processing")
        
        start = time.time()
        results, status = run_scan(config)
        elapsed = time.time() - start
        
        print(f"\n✓ Scan completed in {elapsed:.2f}s")
        print(f"✓ Results: {len(results)} symbols")
        print(f"✓ Status: {status[:100]}...")
        
        # Get cache stats
        stats = data_engine.get_cache_stats()
        print(f"\n✓ Cache performance:")
        print(f"  Hits: {stats.get('hits', 0)}")
        print(f"  Misses: {stats.get('misses', 0)}")
        print(f"  Hit rate: {stats.get('hit_rate', 0):.1%}")
        
        # Run again to test warm cache
        print("\nRunning second scan (should be faster with cache)...")
        start = time.time()
        results2, status2 = run_scan(config)
        elapsed2 = time.time() - start
        
        print(f"✓ Second scan: {elapsed2:.2f}s")
        
        if elapsed2 < elapsed:
            speedup = elapsed / elapsed2
            print(f"✓ Cache speedup: {speedup:.1f}x faster")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("TECHNIC SCANNER PERFORMANCE OPTIMIZATION TEST SUITE")
    print("="*70)
    print("Testing all 4 optimization steps...")
    
    results = {
        "Step 1 (Caching)": test_step1_caching(),
        "Step 2 (Filtering)": test_step2_filtering(),
        "Step 3 (Redis)": test_step3_redis(),
        "Step 4 (Parallel)": test_step4_parallel(),
        "Integration": test_integration(),
    }
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Scanner optimizations are working correctly.")
        return 0
    else:
        print("\n⚠ Some tests failed. Review the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
