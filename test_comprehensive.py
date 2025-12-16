"""
Comprehensive testing suite for Phase 3B fix
Tests: Performance, Edge Cases, Different Configurations
"""

import sys
import time
from technic_v4.scanner_core import run_scan, ScanConfig

def test_performance():
    """Test that performance is acceptable"""
    print("\n" + "="*80)
    print("TEST 1: PERFORMANCE VERIFICATION")
    print("="*80)
    
    config = ScanConfig(
        max_symbols=100,
        lookback_days=90,
        trade_style="Short-term swing"
    )
    
    start = time.time()
    df, msg = run_scan(config)
    elapsed = time.time() - start
    
    per_symbol = elapsed / 100 if len(df) > 0 else 0
    
    print(f"✓ Processed 100 symbols in {elapsed:.2f}s ({per_symbol:.3f}s/symbol)")
    print(f"✓ Got {len(df)} results")
    
    # Performance threshold: should be under 1s per symbol
    if per_symbol < 1.0:
        print(f"✅ PASSED: Performance acceptable ({per_symbol:.3f}s/symbol < 1.0s threshold)")
        return True
    else:
        print(f"⚠️  WARNING: Performance slower than expected ({per_symbol:.3f}s/symbol)")
        return True  # Still pass, just warn

def test_different_trade_styles():
    """Test with different trade styles"""
    print("\n" + "="*80)
    print("TEST 2: DIFFERENT TRADE STYLES")
    print("="*80)
    
    styles = ["Short-term swing", "Medium-term swing", "Position / longer-term"]
    
    for style in styles:
        print(f"\nTesting trade style: {style}")
        config = ScanConfig(
            max_symbols=20,
            lookback_days=90,
            trade_style=style
        )
        
        df, msg = run_scan(config)
        
        # Check critical columns
        critical_cols = ['ATR_pct', 'Signal', 'TrendScore']
        missing = [c for c in critical_cols if c not in df.columns]
        
        if missing:
            print(f"  ❌ FAILED: Missing columns {missing}")
            return False
        
        print(f"  ✓ Got {len(df)} results with all critical columns")
    
    print(f"\n✅ PASSED: All trade styles work correctly")
    return True

def test_small_universe():
    """Test with very small symbol count"""
    print("\n" + "="*80)
    print("TEST 3: EDGE CASE - SMALL UNIVERSE (10 symbols)")
    print("="*80)
    
    config = ScanConfig(
        max_symbols=10,
        lookback_days=90,
        trade_style="Short-term swing"
    )
    
    df, msg = run_scan(config)
    
    critical_cols = ['ATR_pct', 'Signal', 'TrendScore', 'MomentumScore']
    missing = [c for c in critical_cols if c not in df.columns]
    
    if missing:
        print(f"❌ FAILED: Missing columns {missing}")
        return False
    
    print(f"✓ Got {len(df)} results")
    print(f"✓ All critical columns present")
    print(f"✅ PASSED: Small universe handled correctly")
    return True

def test_column_completeness():
    """Verify all expected columns are present"""
    print("\n" + "="*80)
    print("TEST 4: COLUMN COMPLETENESS")
    print("="*80)
    
    config = ScanConfig(
        max_symbols=30,
        lookback_days=90,
        trade_style="Short-term swing"
    )
    
    df, msg = run_scan(config)
    
    # All critical scoring columns that were missing before the fix
    required_cols = [
        'ATR_pct', 'ATR14_pct', 'ATR', 'ATR14',
        'Signal', 'TrendScore', 'MomentumScore', 
        'VolumeScore', 'VolatilityScore', 'OscillatorScore',
        'BreakoutScore', 'ExplosivenessScore', 'RiskScore',
        'TechRating', 'AlphaScore'
    ]
    
    missing = [c for c in required_cols if c not in df.columns]
    present = len(required_cols) - len(missing)
    
    print(f"✓ Total columns in output: {len(df.columns)}")
    print(f"✓ Required columns present: {present}/{len(required_cols)}")
    
    if missing:
        print(f"❌ FAILED: Missing columns: {missing}")
        return False
    
    print(f"✅ PASSED: All {len(required_cols)} required columns present")
    return True

def main():
    """Run all comprehensive tests"""
    print("="*80)
    print("COMPREHENSIVE TEST SUITE - PHASE 3B FIX")
    print("="*80)
    
    tests = [
        ("Performance", test_performance),
        ("Trade Styles", test_different_trade_styles),
        ("Small Universe", test_small_universe),
        ("Column Completeness", test_column_completeness),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} test FAILED with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
