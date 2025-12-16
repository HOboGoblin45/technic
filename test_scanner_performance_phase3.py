#!/usr/bin/env python3
"""
Test scanner performance after Phase 3A implementation
Compare baseline vs optimized performance
"""

import time
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_scanner_performance():
    """Test scanner with different configurations to measure speedup"""
    print("=" * 60)
    print("SCANNER PERFORMANCE TEST - PHASE 3A")
    print("=" * 60)
    
    try:
        from technic_v4.scanner_core import run_scan, ScanConfig
        
        # Test configurations
        test_configs = [
            {"max_symbols": 10, "name": "Small (10 symbols)"},
            {"max_symbols": 50, "name": "Medium (50 symbols)"},
            {"max_symbols": 100, "name": "Large (100 symbols)"},
            {"max_symbols": 500, "name": "Extra Large (500 symbols)"},
        ]
        
        results = []
        
        for test_config in test_configs:
            print(f"\n📊 Testing: {test_config['name']}")
            print("-" * 40)
            
            config = ScanConfig(
                max_symbols=test_config['max_symbols'],
                lookback_days=30,  # Reduced for faster testing
                min_tech_rating=0.0
            )
            
            # Measure scan time
            start_time = time.time()
            
            try:
                df, status = run_scan(config)
                elapsed = time.time() - start_time
                
                if df is not None and not df.empty:
                    results_count = len(df)
                    per_symbol_time = elapsed / test_config['max_symbols']
                    
                    result = {
                        'config': test_config['name'],
                        'max_symbols': test_config['max_symbols'],
                        'results': results_count,
                        'total_time': elapsed,
                        'per_symbol': per_symbol_time,
                        'status': 'SUCCESS'
                    }
                    
                    print(f"✅ Success!")
                    print(f"   Total time: {elapsed:.2f}s")
                    print(f"   Per symbol: {per_symbol_time:.3f}s")
                    print(f"   Results found: {results_count}")
                    print(f"   Status: {status[:100]}...")
                else:
                    result = {
                        'config': test_config['name'],
                        'max_symbols': test_config['max_symbols'],
                        'results': 0,
                        'total_time': elapsed,
                        'per_symbol': elapsed / test_config['max_symbols'],
                        'status': 'NO_RESULTS'
                    }
                    print(f"⚠️ No results returned")
                    
            except Exception as e:
                elapsed = time.time() - start_time
                result = {
                    'config': test_config['name'],
                    'max_symbols': test_config['max_symbols'],
                    'results': 0,
                    'total_time': elapsed,
                    'per_symbol': 0,
                    'status': f'ERROR: {str(e)[:50]}'
                }
                print(f"❌ Error: {e}")
            
            results.append(result)
            
            # Don't overwhelm the API
            if test_config['max_symbols'] < 500:
                time.sleep(2)
        
        # Print summary
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        
        print("\n| Config | Symbols | Time (s) | Per Symbol (s) | Results | Status |")
        print("|--------|---------|----------|----------------|---------|---------|")
        
        for r in results:
            print(f"| {r['config']:20} | {r['max_symbols']:7} | {r['total_time']:8.2f} | {r['per_symbol']:14.4f} | {r['results']:7} | {r['status'][:7]} |")
        
        # Calculate speedup
        if results:
            avg_per_symbol = sum(r['per_symbol'] for r in results if r['per_symbol'] > 0) / len([r for r in results if r['per_symbol'] > 0])
            baseline_per_symbol = 0.613  # From original measurements
            speedup = baseline_per_symbol / avg_per_symbol if avg_per_symbol > 0 else 0
            
            print(f"\n📈 PERFORMANCE METRICS:")
            print(f"   Average per symbol: {avg_per_symbol:.4f}s")
            print(f"   Baseline per symbol: {baseline_per_symbol:.4f}s")
            print(f"   Speedup achieved: {speedup:.1f}x")
            
            # Projection for full universe
            full_universe_time = avg_per_symbol * 5500  # Average of 5000-6000
            print(f"\n🎯 FULL UNIVERSE PROJECTION (5,500 symbols):")
            print(f"   Estimated time: {full_universe_time:.0f} seconds ({full_universe_time/60:.1f} minutes)")
            
            if full_universe_time <= 90:
                print(f"   ✅ TARGET ACHIEVED! Under 90 seconds!")
            else:
                additional_speedup_needed = full_universe_time / 90
                print(f"   ⚠️ Additional {additional_speedup_needed:.1f}x speedup needed for 90-second target")
                print(f"   Next steps: Implement Phase 3B (Redis) and 3C (Ray)")
        
        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print(f"Starting performance test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    test_scanner_performance()
