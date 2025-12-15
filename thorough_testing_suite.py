"""
Thorough Testing Suite for Scanner Performance Optimizations
Covers all edge cases, performance scenarios, and integration tests
"""

import time
import sys
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))

def test_1_full_scan_endpoint():
    """Test 1: Full scan via API endpoint"""
    print("\n" + "="*70)
    print("TEST 1: FULL SCAN ENDPOINT")
    print("="*70)
    
    try:
        from technic_v4.scanner_core import run_scan, ScanConfig
        from technic_v4 import data_engine
        
        # Clear cache for fair test
        data_engine.clear_memory_cache()
        
        # Run a full scan with default settings
        print("Running full scan with default settings...")
        config = ScanConfig(trade_style="swing")
        
        start = time.time()
        results, status = run_scan(config)
        elapsed = time.time() - start
        
        print(f"✓ Scan completed in {elapsed:.2f}s")
        print(f"✓ Results: {len(results)} symbols")
        print(f"✓ Status: {status[:100]}...")
        
        # Verify results structure
        if len(results) > 0:
            first_result = results.iloc[0]
            required_fields = ['Symbol', 'Signal', 'TechRating', 'Entry', 'Stop', 'Target']
            missing = [f for f in required_fields if f not in results.columns]
            if missing:
                print(f"⚠ Missing fields: {missing}")
            else:
                print(f"✓ All required fields present")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
        return False


def test_2_cache_persistence():
    """Test 2: Cache persistence across operations"""
    print("\n" + "="*70)
    print("TEST 2: CACHE PERSISTENCE")
    print("="*70)
    
    try:
        from technic_v4 import data_engine
        
        # Clear and get initial stats
        data_engine.clear_memory_cache()
        initial_stats = data_engine.get_cache_stats()
        print(f"Initial cache: {initial_stats}")
        
        # Fetch some data
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        print(f"\nFetching data for {len(symbols)} symbols...")
        
        for symbol in symbols:
            df = data_engine.get_price_history(symbol, days=90)
            if df is not None:
                print(f"  ✓ {symbol}: {len(df)} bars")
        
        # Check cache stats
        stats = data_engine.get_cache_stats()
        print(f"\n✓ Cache stats after fetch:")
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")
        print(f"  Cache size: {stats['cache_size']}")
        print(f"  Hit rate: {stats['hit_rate']:.1%}")
        
        # Fetch again (should hit cache)
        print(f"\nFetching same symbols again (should hit cache)...")
        start = time.time()
        for symbol in symbols:
            df = data_engine.get_price_history(symbol, days=90)
        elapsed = time.time() - start
        
        stats2 = data_engine.get_cache_stats()
        print(f"✓ Second fetch: {elapsed:.3f}s")
        print(f"✓ Hit rate improved: {stats2['hit_rate']:.1%}")
        
        if stats2['hit_rate'] > stats['hit_rate']:
            print("✓ Cache persistence verified")
            return True
        else:
            print("⚠ Cache hit rate did not improve")
            return False
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
        return False


def test_3_concurrent_scans():
    """Test 3: Multiple concurrent scans"""
    print("\n" + "="*70)
    print("TEST 3: CONCURRENT SCANS")
    print("="*70)
    
    try:
        from technic_v4.scanner_core import run_scan, ScanConfig
        
        def run_single_scan(scan_id):
            """Run a single scan"""
            config = ScanConfig(
                trade_style="swing",
                sectors=["Technology"] if scan_id % 2 == 0 else ["Healthcare"]
            )
            start = time.time()
            results, status = run_scan(config)
            elapsed = time.time() - start
            return scan_id, len(results), elapsed
        
        # Run 3 concurrent scans
        num_scans = 3
        print(f"Running {num_scans} concurrent scans...")
        
        with ThreadPoolExecutor(max_workers=num_scans) as executor:
            futures = [executor.submit(run_single_scan, i) for i in range(num_scans)]
            
            results = []
            for future in as_completed(futures):
                scan_id, num_results, elapsed = future.result()
                results.append((scan_id, num_results, elapsed))
                print(f"  ✓ Scan {scan_id}: {num_results} results in {elapsed:.2f}s")
        
        print(f"\n✓ All {num_scans} scans completed successfully")
        avg_time = sum(r[2] for r in results) / len(results)
        print(f"✓ Average scan time: {avg_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
        return False


def test_4_edge_case_empty_universe():
    """Test 4: Edge case - empty universe after filtering"""
    print("\n" + "="*70)
    print("TEST 4: EDGE CASE - EMPTY UNIVERSE")
    print("="*70)
    
    try:
        from technic_v4.scanner_core import run_scan, ScanConfig
        
        # Create config that should result in empty universe
        config = ScanConfig(
            trade_style="swing",
            sectors=["NonExistentSector123"],  # Invalid sector
        )
        
        print("Running scan with invalid sector filter...")
        results, status = run_scan(config)
        
        if len(results) == 0:
            print("✓ Handled empty universe gracefully")
            print(f"✓ Status message: {status[:100]}...")
            return True
        else:
            print(f"⚠ Expected empty results, got {len(results)}")
            return False
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
        return False


def test_5_edge_case_invalid_symbols():
    """Test 5: Edge case - invalid symbols in data fetch"""
    print("\n" + "="*70)
    print("TEST 5: EDGE CASE - INVALID SYMBOLS")
    print("="*70)
    
    try:
        from technic_v4 import data_engine
        
        invalid_symbols = ['INVALID123', 'NOTREAL', 'FAKE']
        print(f"Testing with invalid symbols: {invalid_symbols}")
        
        for symbol in invalid_symbols:
            df = data_engine.get_price_history(symbol, days=90)
            if df is None or df.empty:
                print(f"  ✓ {symbol}: Handled gracefully (returned None/empty)")
            else:
                print(f"  ⚠ {symbol}: Unexpected data returned")
        
        print("✓ Invalid symbols handled without crashes")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
        return False


def test_6_memory_profiling():
    """Test 6: Memory usage profiling"""
    print("\n" + "="*70)
    print("TEST 6: MEMORY PROFILING")
    print("="*70)
    
    try:
        import psutil
        import os
        from technic_v4 import data_engine
        
        process = psutil.Process(os.getpid())
        
        # Get initial memory
        initial_mem = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory: {initial_mem:.1f} MB")
        
        # Clear cache
        data_engine.clear_memory_cache()
        
        # Fetch data for many symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'INTC', 'CSCO']
        print(f"\nFetching data for {len(symbols)} symbols...")
        
        for symbol in symbols:
            data_engine.get_price_history(symbol, days=90)
        
        # Check memory after caching
        cached_mem = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = cached_mem - initial_mem
        
        print(f"Memory after caching: {cached_mem:.1f} MB")
        print(f"Memory increase: {mem_increase:.1f} MB")
        
        # Clear cache
        data_engine.clear_memory_cache()
        
        # Check memory after clearing
        cleared_mem = process.memory_info().rss / 1024 / 1024  # MB
        mem_freed = cached_mem - cleared_mem
        
        print(f"Memory after clearing: {cleared_mem:.1f} MB")
        print(f"Memory freed: {mem_freed:.1f} MB")
        
        if mem_increase < 500:  # Less than 500MB increase is reasonable
            print(f"✓ Memory usage reasonable ({mem_increase:.1f} MB)")
            return True
        else:
            print(f"⚠ High memory usage ({mem_increase:.1f} MB)")
            return False
        
    except ImportError:
        print("⚠ psutil not installed, skipping memory profiling")
        print("  Install with: pip install psutil")
        return True  # Not a failure, just unavailable
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
        return False


def test_7_filter_effectiveness():
    """Test 7: Smart filter effectiveness"""
    print("\n" + "="*70)
    print("TEST 7: SMART FILTER EFFECTIVENESS")
    print("="*70)
    
    try:
        from technic_v4.scanner_core import _smart_filter_universe, _prepare_universe
        from technic_v4.universe_loader import load_universe
        from dataclasses import dataclass
        
        @dataclass
        class MockConfig:
            sectors: list = None
            subindustries: list = None
            industry_contains: str = None
        
        # Test with no sector filter
        config = MockConfig()
        universe = load_universe()
        
        print(f"Original universe: {len(universe)} symbols")
        
        # Apply smart filter
        filtered = _smart_filter_universe(universe, config)
        reduction = ((len(universe) - len(filtered)) / len(universe) * 100) if len(universe) > 0 else 0
        
        print(f"After smart filter: {len(filtered)} symbols")
        print(f"Reduction: {reduction:.1f}%")
        
        # Test with sector filter
        config2 = MockConfig(sectors=["Technology"])
        filtered2 = _prepare_universe(config2)
        
        print(f"\nWith Technology sector filter: {len(filtered2)} symbols")
        
        # Verify all are Technology
        tech_count = sum(1 for row in filtered2 if row.sector == "Technology")
        print(f"Technology symbols: {tech_count}/{len(filtered2)}")
        
        if tech_count == len(filtered2):
            print("✓ Sector filter working correctly")
            return True
        else:
            print(f"⚠ Some non-Technology symbols present")
            return False
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
        return False


def test_8_parallel_worker_scaling():
    """Test 8: Parallel worker scaling"""
    print("\n" + "="*70)
    print("TEST 8: PARALLEL WORKER SCALING")
    print("="*70)
    
    try:
        from technic_v4 import scanner_core
        import os
        
        # Check worker configuration
        max_workers = scanner_core.MAX_WORKERS
        cpu_count = os.cpu_count() or 4
        expected = min(32, cpu_count * 2)
        
        print(f"CPU cores: {cpu_count}")
        print(f"MAX_WORKERS: {max_workers}")
        print(f"Expected: {expected}")
        
        if max_workers == expected:
            print("✓ Worker count correctly scaled to CPU")
            return True
        else:
            print(f"⚠ Worker count mismatch")
            return False
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
        return False


def test_9_performance_regression():
    """Test 9: Performance regression check"""
    print("\n" + "="*70)
    print("TEST 9: PERFORMANCE REGRESSION CHECK")
    print("="*70)
    
    try:
        from technic_v4.scanner_core import run_scan, ScanConfig
        from technic_v4 import data_engine
        
        # Clear cache
        data_engine.clear_memory_cache()
        
        # Run cold scan
        config = ScanConfig(trade_style="swing", sectors=["Technology"])
        
        print("Running cold scan...")
        start = time.time()
        results1, _ = run_scan(config)
        cold_time = time.time() - start
        
        print(f"Cold scan: {cold_time:.2f}s ({len(results1)} results)")
        
        # Run warm scan
        print("Running warm scan...")
        start = time.time()
        results2, _ = run_scan(config)
        warm_time = time.time() - start
        
        print(f"Warm scan: {warm_time:.2f}s ({len(results2)} results)")
        
        # Check performance targets
        cold_target = 20  # seconds
        warm_target = 10  # seconds
        
        cold_pass = cold_time < cold_target
        warm_pass = warm_time < warm_target
        
        print(f"\n{'✓' if cold_pass else '✗'} Cold scan {'<' if cold_pass else '>'} {cold_target}s target")
        print(f"{'✓' if warm_pass else '✗'} Warm scan {'<' if warm_pass else '>'} {warm_target}s target")
        
        if warm_time < cold_time:
            speedup = cold_time / warm_time if warm_time > 0 else float('inf')
            print(f"✓ Cache speedup: {speedup:.1f}x")
        
        return cold_pass and warm_pass
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
        return False


def test_10_integration_full_workflow():
    """Test 10: Full workflow integration"""
    print("\n" + "="*70)
    print("TEST 10: FULL WORKFLOW INTEGRATION")
    print("="*70)
    
    try:
        from technic_v4.scanner_core import run_scan, ScanConfig
        from technic_v4 import data_engine
        
        print("Step 1: Clear cache")
        data_engine.clear_memory_cache()
        
        print("Step 2: Run scan")
        config = ScanConfig(trade_style="swing", sectors=["Technology"])
        results, status = run_scan(config)
        print(f"  ✓ Got {len(results)} results")
        
        print("Step 3: Verify cache populated")
        stats = data_engine.get_cache_stats()
        print(f"  ✓ Cache size: {stats['cache_size']}")
        
        print("Step 4: Run second scan (should be faster)")
        start = time.time()
        results2, _ = run_scan(config)
        elapsed = time.time() - start
        print(f"  ✓ Second scan: {elapsed:.2f}s")
        
        print("Step 5: Verify results consistency")
        if len(results) == len(results2):
            print(f"  ✓ Consistent results ({len(results)} symbols)")
        else:
            print(f"  ⚠ Result count mismatch: {len(results)} vs {len(results2)}")
        
        print("\n✓ Full workflow integration successful")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all thorough tests"""
    print("\n" + "="*70)
    print("THOROUGH TESTING SUITE - SCANNER PERFORMANCE OPTIMIZATIONS")
    print("="*70)
    print("Running comprehensive tests (this may take several minutes)...")
    
    tests = [
        ("Full Scan Endpoint", test_1_full_scan_endpoint),
        ("Cache Persistence", test_2_cache_persistence),
        ("Concurrent Scans", test_3_concurrent_scans),
        ("Edge Case: Empty Universe", test_4_edge_case_empty_universe),
        ("Edge Case: Invalid Symbols", test_5_edge_case_invalid_symbols),
        ("Memory Profiling", test_6_memory_profiling),
        ("Filter Effectiveness", test_7_filter_effectiveness),
        ("Parallel Worker Scaling", test_8_parallel_worker_scaling),
        ("Performance Regression", test_9_performance_regression),
        ("Full Workflow Integration", test_10_integration_full_workflow),
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results[test_name] = False
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*70)
    print("THOROUGH TESTING SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print(f"Time: {total_time:.1f} seconds")
    
    if passed == total:
        print("\n🎉 ALL THOROUGH TESTS PASSED!")
        return 0
    else:
        print("\n⚠ Some tests failed. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
