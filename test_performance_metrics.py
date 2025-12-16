"""
Test performance metrics for Phase 3D-A Task 3
"""

import os
os.environ['REDIS_URL'] = 'redis://:ytvZ1OVXoGV4OenJH3GJYkDLeg2emqad@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0'

from technic_v4.scanner_core import run_scan, ScanConfig

print("="*80)
print("TESTING PERFORMANCE METRICS")
print("="*80)

# Run small scan to test performance metrics
config = ScanConfig(
    max_symbols=10,
    lookback_days=90,
    trade_style="Short-term swing"
)

print("\n[TEST] Running scan to collect performance metrics...")
print()

result = run_scan(config)

# Check if we got 3 return values (new format) or 2 (old format)
if len(result) == 3:
    df, msg, metrics = result
    print("\n✅ New format detected: 3 return values (df, msg, metrics)")
    
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    
    print(f"\n📊 Scan Performance:")
    print(f"  Total Time: {metrics['total_seconds']:.2f}s")
    print(f"  Symbols Scanned: {metrics['symbols_scanned']}")
    print(f"  Symbols Returned: {metrics['symbols_returned']}")
    print(f"  Speed: {metrics['symbols_per_second']:.2f} symbols/second")
    print(f"  Speedup: {metrics['speedup']:.1f}x vs baseline")
    print(f"  Baseline Time: {metrics['baseline_time']:.1f}s")
    
    if 'cache_available' in metrics:
        print(f"\n💾 Cache Performance:")
        print(f"  Cache Available: {'✅ Yes' if metrics['cache_available'] else '❌ No'}")
        if metrics['cache_available']:
            print(f"  Hit Rate: {metrics['cache_hit_rate']:.1f}%")
            print(f"  Cache Hits: {metrics['cache_hits']}")
            print(f"  Cache Misses: {metrics['cache_misses']}")
            print(f"  Total Keys: {metrics['total_keys']}")
    
    print(f"\n📈 Results:")
    print(f"  Status: {msg}")
    print(f"  Results Count: {len(df)}")
    
    # Validate metrics structure
    print("\n" + "="*80)
    print("METRICS VALIDATION")
    print("="*80)
    
    required_keys = [
        'total_seconds',
        'symbols_scanned',
        'symbols_returned',
        'symbols_per_second',
        'speedup',
        'baseline_time'
    ]
    
    all_present = True
    for key in required_keys:
        present = key in metrics
        status = "✅" if present else "❌"
        print(f"  {status} {key}: {metrics.get(key, 'MISSING')}")
        if not present:
            all_present = False
    
    if all_present:
        print("\n✅ All required metrics present!")
    else:
        print("\n❌ Some metrics missing!")
    
    # Check metric values are reasonable
    print("\n" + "="*80)
    print("METRICS SANITY CHECKS")
    print("="*80)
    
    checks = []
    checks.append(("Total time > 0", metrics['total_seconds'] > 0))
    checks.append(("Symbols scanned > 0", metrics['symbols_scanned'] > 0))
    checks.append(("Speed > 0", metrics['symbols_per_second'] > 0))
    checks.append(("Speedup >= 1", metrics['speedup'] >= 1.0))
    checks.append(("Baseline time > actual time", metrics['baseline_time'] >= metrics['total_seconds']))
    
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
    
    all_passed = all(passed for _, passed in checks)
    
    if all_passed:
        print("\n✅ All sanity checks passed!")
    else:
        print("\n⚠️  Some sanity checks failed!")
    
else:
    df, msg = result
    print("\n⚠️  Old format detected: 2 return values (df, msg)")
    print("   Performance metrics not available")
    print(f"\n  Status: {msg}")
    print(f"  Results: {len(df)}")

print("\n" + "="*80)
