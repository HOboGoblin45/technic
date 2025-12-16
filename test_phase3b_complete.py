#!/usr/bin/env python3
"""Comprehensive Phase 3B testing suite"""

import time
import pandas as pd
import sys
from technic_v4.scanner_core import ScanConfig, run_scan

def test_small_scan():
    """Test 100 symbols"""
    print("\n" + "=" * 80)
    print("TEST 1: Small Scan (100 symbols)")
    print("=" * 80)
    
    config = ScanConfig(max_symbols=100, lookback_days=150)
    start = time.time()
    
    try:
        results, _ = run_scan(config)
        elapsed = time.time() - start
        
        print(f"Time: {elapsed:.2f}s")
        print(f"Results: {len(results)}")
        print(f"Per-symbol: {elapsed/max(len(results), 1):.3f}s")
        
        if elapsed >= 60:
            print(f"⚠️  WARNING: {elapsed:.2f}s > 60s target")
            return False
        
        if len(results) == 0:
            print("❌ FAILED: No results")
            return False
            
        print("✅ PASSED")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_medium_scan():
    """Test 1000 symbols"""
    print("\n" + "=" * 80)
    print("TEST 2: Medium Scan (1000 symbols)")
    print("=" * 80)
    
    config = ScanConfig(max_symbols=1000, lookback_days=150)
    start = time.time()
    
    try:
        results, _ = run_scan(config)
        elapsed = time.time() - start
        
        print(f"Time: {elapsed:.2f}s ({elapsed/60:.1f} minutes)")
        print(f"Results: {len(results)}")
        print(f"Per-symbol: {elapsed/max(len(results), 1):.3f}s")
        
        # Target: <5 minutes
        if elapsed >= 300:
            print(f"⚠️  WARNING: {elapsed:.2f}s > 300s (5 min) target")
            return False
            
        print("✅ PASSED")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_result_quality():
    """Verify result quality"""
    print("\n" + "=" * 80)
    print("TEST 3: Result Quality")
    print("=" * 80)
    
    config = ScanConfig(max_symbols=50, lookback_days=150)
    
    try:
        results, _ = run_scan(config)
        
        # Check required columns
        required = ['Symbol', 'TechRating', 'Signal', 'Close', 'Volume', 
                    'RSI', 'MACD', 'ATR_pct']
        missing = [col for col in required if col not in results.columns]
        
        if missing:
            print(f"❌ FAILED: Missing columns: {missing}")
            return False
        
        # Check for NaN values in critical columns
        for col in ['TechRating', 'Close']:
            nan_count = results[col].isna().sum()
            if nan_count > 0:
                print(f"❌ FAILED: {col} has {nan_count} NaN values")
                return False
        
        print(f"✅ All {len(required)} required columns present")
        print(f"✅ No NaN values in critical columns")
        print(f"✅ Sample TechRating range: {results['TechRating'].min():.1f} - {results['TechRating'].max():.1f}")
        print("✅ PASSED")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_full_scan():
    """Test 3000 symbols (reduced from 5000 for faster testing)"""
    print("\n" + "=" * 80)
    print("TEST 4: Large Scan (3000 symbols)")
    print("=" * 80)
    
    config = ScanConfig(max_symbols=3000, lookback_days=150)
    start = time.time()
    
    try:
        results, _ = run_scan(config)
        elapsed = time.time() - start
        
        print(f"Time: {elapsed:.2f}s ({elapsed/60:.1f} minutes)")
        print(f"Results: {len(results)}")
        print(f"Per-symbol: {elapsed/max(len(results), 1):.3f}s")
        
        # Target: <6 minutes for 3000 symbols
        if elapsed >= 360:
            print(f"⚠️  WARNING: {elapsed:.2f}s > 360s (6 min) target")
            print(f"⚠️  Extrapolated 5000 symbols: {(elapsed/3000)*5000/60:.1f} minutes")
            return False
        
        # Extrapolate to 5000 symbols
        extrapolated = (elapsed / 3000) * 5000
        print(f"✅ Extrapolated 5000 symbols: {extrapolated/60:.1f} minutes")
        
        if extrapolated <= 360:
            print(f"✅ PASSED - Phase 3B Target Achieved!")
            return True
        else:
            print(f"⚠️  Extrapolated time exceeds 6 minute target")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("PHASE 3B COMPREHENSIVE TEST SUITE")
    print("Goal: 3-6 minute scans (2-3x speedup)")
    print("=" * 80)
    
    tests = [
        ("Small Scan (100)", test_small_scan),
        ("Medium Scan (1000)", test_medium_scan),
        ("Result Quality", test_result_quality),
        ("Large Scan (3000)", test_full_scan),
    ]
    
    passed = 0
    failed = 0
    results_summary = []
    
    for name, test_func in tests:
        print(f"\nRunning: {name}...")
        try:
            if test_func():
                passed += 1
                results_summary.append(f"✅ {name}")
            else:
                failed += 1
                results_summary.append(f"❌ {name}")
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1
            results_summary.append(f"❌ {name} (Exception)")
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for result in results_summary:
        print(result)
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    print("=" * 80)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED - PHASE 3B COMPLETE!")
        print("✅ Vectorization & Ray optimization working")
        print("✅ Ready to proceed to Phase 3C (Redis Caching)")
        print("\nNext steps:")
        print("1. Deploy to Render")
        print("2. Monitor production performance")
        print("3. Begin Phase 3C implementation")
    else:
        print(f"\n⚠️  {failed} TEST(S) FAILED")
        print("Review errors above and fix issues before proceeding")
    
    return failed == 0

if __name__ == "__main__":
    print("\n🚀 Phase 3B: Complete Test Suite\n")
    success = main()
    exit(0 if success else 1)
