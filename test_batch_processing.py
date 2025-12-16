#!/usr/bin/env python3
"""Test batch processing implementation"""

import time
import sys
from technic_v4.scanner_core import ScanConfig, run_scan

def test_batch_processing():
    """Test batch processing with 100 symbols"""
    
    print("=" * 80)
    print("BATCH PROCESSING TEST - Phase 3B")
    print("=" * 80)
    
    # Create config for 100 symbols
    config = ScanConfig(
        max_symbols=100,
        lookback_days=150,
        sectors=["Technology", "Healthcare", "Financials"]
    )
    
    print(f"\nConfiguration:")
    print(f"  Max symbols: {config.max_symbols}")
    print(f"  Lookback days: {config.lookback_days}")
    print(f"  Sectors: {config.sectors}")
    
    # Run scan
    print(f"\nStarting scan...")
    start = time.time()
    
    try:
        results, status = run_scan(config)
        elapsed = time.time() - start
        
        # Verify results
        print(f"\n{'='*80}")
        print("RESULTS:")
        print(f"{'='*80}")
        print(f"✅ Scan completed in {elapsed:.2f} seconds")
        print(f"✅ Results: {len(results)} symbols")
        print(f"✅ Per-symbol: {elapsed/max(len(results), 1):.3f}s")
        print(f"✅ Status: {status}")
        
        # Check for required columns
        required_cols = ['Symbol', 'TechRating', 'Signal', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in results.columns]
        
        if missing:
            print(f"\n❌ Missing columns: {missing}")
            return False
        
        print(f"✅ All required columns present: {required_cols}")
        
        # Show sample results
        if len(results) > 0:
            print(f"\nSample results (top 5):")
            print(results[['Symbol', 'TechRating', 'Signal', 'Close']].head())
        
        # Target: <60 seconds for 100 symbols (0.6s per symbol)
        target_time = 60
        print(f"\n{'='*80}")
        print("PERFORMANCE CHECK:")
        print(f"{'='*80}")
        print(f"Target: <{target_time}s for 100 symbols")
        print(f"Actual: {elapsed:.2f}s")
        
        if elapsed <= target_time:
            print(f"✅ PASSED: {elapsed:.2f}s <= {target_time}s target")
            print(f"✅ Speedup achieved!")
            return True
        else:
            print(f"⚠️  SLOW: {elapsed:.2f}s > {target_time}s target")
            print(f"⚠️  Need further optimization")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🚀 Phase 3B: Batch Processing Test\n")
    success = test_batch_processing()
    
    if success:
        print("\n" + "="*80)
        print("🎉 TEST PASSED - Batch processing working correctly!")
        print("="*80)
        exit(0)
    else:
        print("\n" + "="*80)
        print("❌ TEST FAILED - Review errors above")
        print("="*80)
        exit(1)
