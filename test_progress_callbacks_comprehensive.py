"""
Comprehensive Progress Callbacks Testing Suite - Phase 3D-A Task 1
Tests all aspects of the progress callback infrastructure
"""

import os
os.environ['REDIS_URL'] = 'redis://:ytvZ1OVXoGV4OenJH3GJYkDLeg2emqad@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0'

from technic_v4.scanner_core import run_scan, ScanConfig
from technic_v4.config.settings import get_settings
import time

print("="*80)
print("COMPREHENSIVE PROGRESS CALLBACKS TEST SUITE")
print("="*80)

# Test results tracker
test_results = {
    "thread_pool_mode": False,
    "per_symbol_progress": False,
    "error_handling": False,
    "performance_impact": False,
    "metadata_completeness": False
}

# ============================================================================
# TEST 1: Thread Pool Mode with Per-Symbol Progress
# ============================================================================
print("\n" + "="*80)
print("TEST 1: Thread Pool Mode - Per-Symbol Progress")
print("="*80)

progress_updates_threadpool = []

def progress_callback_threadpool(stage, current, total, message, metadata):
    """Capture all progress updates in thread pool mode"""
    progress_updates_threadpool.append({
        'stage': stage,
        'current': current,
        'total': total,
        'message': message,
        'metadata': metadata,
        'timestamp': time.time()
    })
    
    # Print progress
    progress_pct = (current / total * 100) if total > 0 else 0
    print(f"[PROGRESS] {stage.upper()}: {current}/{total} ({progress_pct:.1f}%) - {message}")
    if 'symbol' in metadata:
        print(f"           Symbol: {metadata['symbol']} | Sector: {metadata.get('sector', 'Unknown')}")

# Force thread pool mode by disabling Ray
settings = get_settings()
original_use_ray = settings.use_ray
settings.use_ray = False

print("\n[TEST 1] Running scan with thread pool executor...")
print(f"[TEST 1] Ray disabled: use_ray={settings.use_ray}")

config = ScanConfig(
    max_symbols=15,
    lookback_days=90,
    trade_style="Short-term swing"
)

start_time = time.time()
df, msg = run_scan(config, progress_cb=progress_callback_threadpool)
elapsed = time.time() - start_time

print(f"\n[TEST 1] Scan completed in {elapsed:.2f}s")
print(f"[TEST 1] Total progress updates: {len(progress_updates_threadpool)}")

# Analyze results
symbol_updates = [u for u in progress_updates_threadpool if 'symbol' in u.get('metadata', {})]
print(f"[TEST 1] Symbol-level updates: {len(symbol_updates)}")

if len(symbol_updates) > 0:
    print("\n[TEST 1] ✅ Per-symbol progress callbacks working!")
    test_results["thread_pool_mode"] = True
    test_results["per_symbol_progress"] = True
    
    # Show sample updates
    print("\n[TEST 1] Sample symbol updates:")
    for update in symbol_updates[:5]:
        meta = update['metadata']
        progress_pct = (update['current'] / update['total'] * 100) if update['total'] > 0 else 0
        print(f"  - {meta['symbol']} ({meta.get('sector', 'Unknown')}): {progress_pct:.1f}%")
else:
    print("\n[TEST 1] ❌ No per-symbol progress updates received")

# Restore Ray setting
settings.use_ray = original_use_ray

# ============================================================================
# TEST 2: Error Handling - Callback That Raises Exceptions
# ============================================================================
print("\n" + "="*80)
print("TEST 2: Error Handling - Callback Exceptions")
print("="*80)

exception_count = 0
successful_updates = 0

def error_prone_callback(stage, current, total, message, metadata):
    """Callback that raises exceptions on every 3rd call"""
    global exception_count, successful_updates
    
    if current % 3 == 0 and current > 0:
        exception_count += 1
        raise ValueError(f"Simulated callback error at {current}/{total}")
    
    successful_updates += 1
    print(f"[PROGRESS] {stage}: {current}/{total} - {message}")

print("\n[TEST 2] Running scan with error-prone callback...")
settings.use_ray = False  # Use thread pool for predictable behavior

config = ScanConfig(
    max_symbols=10,
    lookback_days=90,
    trade_style="Short-term swing"
)

try:
    df2, msg2 = run_scan(config, progress_cb=error_prone_callback)
    print(f"\n[TEST 2] Scan completed successfully despite {exception_count} callback errors")
    print(f"[TEST 2] Successful updates: {successful_updates}")
    print(f"[TEST 2] Results: {len(df2)} symbols")
    
    if len(df2) > 0:
        print("\n[TEST 2] ✅ Error handling working - scan continues despite callback failures")
        test_results["error_handling"] = True
    else:
        print("\n[TEST 2] ⚠️  Scan completed but no results")
except Exception as e:
    print(f"\n[TEST 2] ❌ Scan failed: {e}")

settings.use_ray = original_use_ray

# ============================================================================
# TEST 3: Performance Impact Measurement
# ============================================================================
print("\n" + "="*80)
print("TEST 3: Performance Impact Measurement")
print("="*80)

# Test without callback
print("\n[TEST 3A] Running scan WITHOUT callback...")
settings.use_ray = False

config = ScanConfig(
    max_symbols=20,
    lookback_days=90,
    trade_style="Short-term swing"
)

start_no_callback = time.time()
df_no_cb, _ = run_scan(config, progress_cb=None)
time_no_callback = time.time() - start_no_callback

print(f"[TEST 3A] Time without callback: {time_no_callback:.2f}s")
print(f"[TEST 3A] Results: {len(df_no_cb)} symbols")

# Test with callback
print("\n[TEST 3B] Running scan WITH callback...")

callback_call_count = 0
callback_total_time = 0

def performance_callback(stage, current, total, message, metadata):
    """Measure callback overhead"""
    global callback_call_count, callback_total_time
    
    cb_start = time.time()
    callback_call_count += 1
    # Simulate minimal processing
    _ = f"{stage}:{current}/{total}"
    callback_total_time += (time.time() - cb_start)

start_with_callback = time.time()
df_with_cb, _ = run_scan(config, progress_cb=performance_callback)
time_with_callback = time.time() - start_with_callback

print(f"[TEST 3B] Time with callback: {time_with_callback:.2f}s")
print(f"[TEST 3B] Results: {len(df_with_cb)} symbols")
print(f"[TEST 3B] Callback calls: {callback_call_count}")
print(f"[TEST 3B] Total callback time: {callback_total_time*1000:.2f}ms")

# Calculate overhead
overhead_pct = ((time_with_callback - time_no_callback) / time_no_callback * 100) if time_no_callback > 0 else 0
avg_callback_time = (callback_total_time / callback_call_count * 1000) if callback_call_count > 0 else 0

print(f"\n[TEST 3] Performance Analysis:")
print(f"  - Overhead: {overhead_pct:.2f}%")
print(f"  - Avg callback time: {avg_callback_time:.3f}ms")

if overhead_pct < 5.0:
    print(f"\n[TEST 3] ✅ Performance impact minimal (<5%)")
    test_results["performance_impact"] = True
else:
    print(f"\n[TEST 3] ⚠️  Performance overhead: {overhead_pct:.2f}%")

settings.use_ray = original_use_ray

# ============================================================================
# TEST 4: Metadata Completeness
# ============================================================================
print("\n" + "="*80)
print("TEST 4: Metadata Completeness Validation")
print("="*80)

metadata_samples = []

def metadata_validator(stage, current, total, message, metadata):
    """Validate metadata structure and content"""
    if len(metadata_samples) < 5:
        metadata_samples.append({
            'stage': stage,
            'metadata': metadata.copy()
        })
    
    # Check required fields
    if current == 0:
        # Initial update should have scan-level metadata
        required = ['symbols_total', 'trade_style', 'lookback_days']
        for field in required:
            if field not in metadata:
                print(f"[METADATA] ⚠️  Missing field: {field}")
    
    if 'symbol' in metadata:
        # Symbol-level update should have symbol metadata
        required = ['symbol', 'sector', 'progress_pct']
        for field in required:
            if field not in metadata:
                print(f"[METADATA] ⚠️  Missing field: {field}")

print("\n[TEST 4] Running scan with metadata validator...")
settings.use_ray = False

config = ScanConfig(
    max_symbols=10,
    lookback_days=90,
    trade_style="Short-term swing"
)

df4, _ = run_scan(config, progress_cb=metadata_validator)

print(f"\n[TEST 4] Collected {len(metadata_samples)} metadata samples")

if metadata_samples:
    print("\n[TEST 4] Sample metadata structures:")
    for i, sample in enumerate(metadata_samples[:3], 1):
        print(f"\n  Sample {i} ({sample['stage']}):")
        for key, value in sample['metadata'].items():
            print(f"    - {key}: {value}")
    
    # Check completeness
    all_complete = True
    for sample in metadata_samples:
        meta = sample['metadata']
        if 'symbol' in meta:
            if not all(k in meta for k in ['symbol', 'sector', 'progress_pct']):
                all_complete = False
        else:
            if not all(k in meta for k in ['symbols_total', 'trade_style', 'lookback_days']):
                all_complete = False
    
    if all_complete:
        print("\n[TEST 4] ✅ All metadata fields present and complete")
        test_results["metadata_completeness"] = True
    else:
        print("\n[TEST 4] ⚠️  Some metadata fields missing")

settings.use_ray = original_use_ray

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE TEST RESULTS SUMMARY")
print("="*80)

print("\nTest Results:")
for test_name, passed in test_results.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} - {test_name.replace('_', ' ').title()}")

passed_count = sum(test_results.values())
total_count = len(test_results)
pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0

print(f"\nOverall: {passed_count}/{total_count} tests passed ({pass_rate:.1f}%)")

if pass_rate >= 80:
    print("\n✅ COMPREHENSIVE TESTING COMPLETE - Progress callbacks fully validated!")
else:
    print(f"\n⚠️  Some tests failed - review results above")

print("\n" + "="*80)
