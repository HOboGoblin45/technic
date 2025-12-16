"""
Quick test to verify Phase 3B optimized Ray runner fix
"""

import sys
import time
from technic_v4.scanner_core import run_scan, ScanConfig

def test_phase3b_fix():
    """Test that optimized Ray runner returns results"""
    
    print("=" * 80)
    print("PHASE 3B FIX VERIFICATION TEST")
    print("=" * 80)
    
    # Small test (50 symbols)
    config = ScanConfig(
        max_symbols=50,
        lookback_days=90,
        trade_style="Short-term swing"
    )
    
    print("\n[TEST] Running scan with 50 symbols...")
    print("[TEST] This should use Phase 3B optimized Ray runner")
    print()
    
    start = time.time()
    df, msg = run_scan(config)
    elapsed = time.time() - start
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Time: {elapsed:.2f}s")
    print(f"Results: {len(df)}")
    print(f"Status: {msg}")
    
    # Check for required columns (scoring columns that were missing before the fix)
    required_cols = ['Symbol', 'TechRating', 'Signal', 'Close', 'Volume', 'RSI', 'ATR_pct', 'TrendScore', 'MomentumScore']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"\n❌ FAILED: Missing columns: {missing}")
        print(f"Available columns: {list(df.columns[:20])}")
        return False
    
    if len(df) == 0:
        print("\n❌ FAILED: No results returned")
        return False
    
    print(f"\n✅ PASSED: Got {len(df)} results with all required columns")
    print(f"\nSample results:")
    print(df[['Symbol', 'TechRating', 'Signal', 'Close']].head())
    
    return True

if __name__ == "__main__":
    success = test_phase3b_fix()
    sys.exit(0 if success else 1)
