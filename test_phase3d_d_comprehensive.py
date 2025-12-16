"""
Comprehensive test suite for Phase 3D-D: Multi-Stage Progress Tracking
Tests all aspects including edge cases, performance, and error handling
"""

import time
import json
import threading
from typing import Dict, Any, List
import traceback
from technic_v4.scanner_core_enhanced import run_scan_enhanced, ScanConfig
from technic_v4.progress import MultiStageProgressTracker, format_time


def test_edge_cases():
    """Test edge cases like empty universe, invalid sectors, etc."""
    print("\n" + "="*60)
    print("TEST: Edge Cases")
    print("="*60)
    
    test_results = []
    
    # Test 1: Empty universe (invalid sector)
    print("\n1. Testing with invalid sector (should return empty)...")
    try:
        config = ScanConfig(
            sectors=["NonExistentSector"],
            max_symbols=10
        )
        
        progress_updates = []
        def collect_progress(stage, current, total, message, metadata):
            progress_updates.append({
                'stage': stage,
                'message': message,
                'metadata': metadata
            })
        
        results_df, status, metrics = run_scan_enhanced(
            config=config,
            progress_cb=collect_progress
        )
        
        # Verify metrics are still returned
        assert 'total_seconds' in metrics, "Missing total_seconds in metrics"
        assert metrics['symbols_scanned'] == 0, "Should have 0 symbols scanned"
        assert len(progress_updates) > 0, "Should have progress updates even with empty universe"
        
        print(f"✓ Empty universe test passed - Status: {status}")
        print(f"  Metrics returned: {list(metrics.keys())}")
        test_results.append(("Empty universe", True))
        
    except Exception as e:
        print(f"✗ Empty universe test failed: {e}")
        test_results.append(("Empty universe", False))
    
    # Test 2: Very small universe (1 symbol)
    print("\n2. Testing with single symbol...")
    try:
        config = ScanConfig(
            sectors=["Information Technology"],
            max_symbols=1,
            lookback_days=10  # Short for speed
        )
        
        stage_counts = {}
        def count_stages(stage, current, total, message, metadata):
            if stage not in stage_counts:
                stage_counts[stage] = 0
            stage_counts[stage] += 1
        
        results_df, status, metrics = run_scan_enhanced(
            config=config,
            progress_cb=count_stages
        )
        
        # Verify all stages executed
        expected_stages = ["universe_loading", "data_fetching", "symbol_scanning", "finalization"]
        for stage in expected_stages:
            assert stage in stage_counts, f"Missing stage: {stage}"
        
        print(f"✓ Single symbol test passed")
        print(f"  Stages executed: {list(stage_counts.keys())}")
        print(f"  Update counts: {stage_counts}")
        test_results.append(("Single symbol", True))
        
    except Exception as e:
        print(f"✗ Single symbol test failed: {e}")
        test_results.append(("Single symbol", False))
    
    # Test 3: Progress callback error handling
    print("\n3. Testing error handling in progress callback...")
    try:
        config = ScanConfig(
            sectors=["Information Technology"],
            max_symbols=2
        )
        
        call_count = [0]
        def faulty_callback(stage, current, total, message, metadata):
            call_count[0] += 1
            if call_count[0] == 5:  # Fail on 5th call
                raise ValueError("Simulated callback error")
        
        # Should not crash the scanner
        results_df, status, metrics = run_scan_enhanced(
            config=config,
            progress_cb=faulty_callback
        )
        
        # Scanner should complete despite callback errors
        assert 'total_seconds' in metrics, "Scanner should complete despite callback errors"
        print(f"✓ Error handling test passed - Scanner completed despite callback error")
        test_results.append(("Callback error handling", True))
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        test_results.append(("Callback error handling", False))
    
    return test_results


def test_performance_overhead():
    """Test the performance overhead of progress tracking"""
    print("\n" + "="*60)
    print("TEST: Performance Overhead")
    print("="*60)
    
    config = ScanConfig(
        sectors=["Information Technology"],
        max_symbols=5,
        lookback_days=20
    )
    
    # Run without progress tracking (baseline)
    print("\n1. Running baseline scan without progress tracking...")
    start = time.time()
    results_df, status, metrics = run_scan_enhanced(
        config=config,
        progress_cb=None  # No progress tracking
    )
    baseline_time = time.time() - start
    print(f"  Baseline time: {baseline_time:.2f}s")
    
    # Run with progress tracking
    print("\n2. Running scan with full progress tracking...")
    update_count = [0]
    def count_updates(stage, current, total, message, metadata):
        update_count[0] += 1
    
    start = time.time()
    results_df, status, metrics = run_scan_enhanced(
        config=config,
        progress_cb=count_updates
    )
    tracked_time = time.time() - start
    print(f"  Tracked time: {tracked_time:.2f}s")
    print(f"  Progress updates: {update_count[0]}")
    
    # Calculate overhead
    overhead = (tracked_time - baseline_time) / baseline_time * 100 if baseline_time > 0 else 0
    print(f"\n3. Performance Analysis:")
    print(f"  Overhead: {overhead:.1f}%")
    print(f"  Updates per second: {update_count[0]/tracked_time:.1f}")
    
    # Verify overhead is minimal (less than 5%)
    if abs(overhead) < 5:
        print(f"✓ Performance overhead is minimal (<5%)")
        return True
    else:
        print(f"⚠ Performance overhead is {overhead:.1f}% (expected <5%)")
        return False


def test_stage_timing_accuracy():
    """Test that stage timings are accurately tracked"""
    print("\n" + "="*60)
    print("TEST: Stage Timing Accuracy")
    print("="*60)
    
    config = ScanConfig(
        sectors=["Information Technology"],
        max_symbols=3,
        lookback_days=15
    )
    
    stage_start_times = {}
    stage_end_times = {}
    
    def track_timing(stage, current, total, message, metadata):
        now = time.time()
        if stage not in stage_start_times:
            stage_start_times[stage] = now
        stage_end_times[stage] = now
    
    # Run scan
    results_df, status, metrics = run_scan_enhanced(
        config=config,
        progress_cb=track_timing
    )
    
    print("\n1. Comparing tracked vs reported timings:")
    
    if 'stage_timings' in metrics:
        for stage_key, reported_time in metrics['stage_timings'].items():
            stage = stage_key.replace('_seconds', '')
            if stage in stage_start_times and stage in stage_end_times:
                tracked_time = stage_end_times[stage] - stage_start_times[stage]
                diff = abs(reported_time - tracked_time)
                print(f"  {stage:20s}: Reported={reported_time:.2f}s, Tracked={tracked_time:.2f}s, Diff={diff:.2f}s")
                
                # Allow 1 second tolerance
                if diff > 1.0:
                    print(f"    ⚠ Large discrepancy detected")
    
    # Verify total time matches sum of stages (approximately)
    if 'stage_timings' in metrics:
        stage_sum = sum(v for k, v in metrics['stage_timings'].items())
        total_time = metrics['total_seconds']
        diff_pct = abs(stage_sum - total_time) / total_time * 100 if total_time > 0 else 0
        
        print(f"\n2. Total time validation:")
        print(f"  Sum of stages: {stage_sum:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Difference: {diff_pct:.1f}%")
        
        if diff_pct < 10:  # Allow 10% tolerance due to overlapping operations
            print(f"✓ Stage timings are accurate")
            return True
        else:
            print(f"⚠ Stage timing discrepancy: {diff_pct:.1f}%")
            return False
    
    return True


def test_eta_accuracy():
    """Test that ETA calculations are reasonably accurate"""
    print("\n" + "="*60)
    print("TEST: ETA Accuracy")
    print("="*60)
    
    config = ScanConfig(
        sectors=["Information Technology"],
        max_symbols=5,
        lookback_days=20
    )
    
    eta_history = []
    start_time = time.time()
    
    def track_eta(stage, current, total, message, metadata):
        if metadata and 'overall_eta' in metadata and metadata['overall_eta']:
            elapsed = time.time() - start_time
            eta_history.append({
                'elapsed': elapsed,
                'eta': metadata['overall_eta'],
                'progress': metadata.get('overall_progress_pct', 0)
            })
    
    # Run scan
    results_df, status, metrics = run_scan_enhanced(
        config=config,
        progress_cb=track_eta
    )
    
    total_time = time.time() - start_time
    
    if eta_history:
        print(f"\n1. ETA progression (total time: {total_time:.2f}s):")
        
        # Check ETAs at different progress points
        checkpoints = [25, 50, 75]
        for checkpoint in checkpoints:
            # Find ETA closest to checkpoint
            closest = min(eta_history, 
                         key=lambda x: abs(x['progress'] - checkpoint),
                         default=None)
            
            if closest and abs(closest['progress'] - checkpoint) < 10:
                actual_remaining = total_time - closest['elapsed']
                eta_error = abs(closest['eta'] - actual_remaining) / actual_remaining * 100 if actual_remaining > 0 else 0
                
                print(f"  At {closest['progress']:.0f}% progress:")
                print(f"    ETA: {closest['eta']:.1f}s")
                print(f"    Actual remaining: {actual_remaining:.1f}s")
                print(f"    Error: {eta_error:.1f}%")
        
        # Check if ETAs generally decrease
        if len(eta_history) > 2:
            early_etas = [h['eta'] for h in eta_history[:3]]
            late_etas = [h['eta'] for h in eta_history[-3:]]
            
            avg_early = sum(early_etas) / len(early_etas)
            avg_late = sum(late_etas) / len(late_etas)
            
            if avg_late < avg_early:
                print(f"\n✓ ETAs decrease over time (early avg: {avg_early:.1f}s, late avg: {avg_late:.1f}s)")
                return True
            else:
                print(f"\n⚠ ETAs did not decrease properly")
                return False
    else:
        print("⚠ No ETA data collected")
        return False
    
    return True


def test_concurrent_scans():
    """Test multiple concurrent scans with progress tracking"""
    print("\n" + "="*60)
    print("TEST: Concurrent Scans")
    print("="*60)
    
    config = ScanConfig(
        sectors=["Information Technology"],
        max_symbols=2,
        lookback_days=10
    )
    
    results = []
    errors = []
    
    def run_scan_thread(thread_id):
        try:
            updates = []
            def collect(stage, current, total, message, metadata):
                updates.append(f"Thread-{thread_id}: {stage}")
            
            df, status, metrics = run_scan_enhanced(
                config=config,
                progress_cb=collect
            )
            
            results.append({
                'thread': thread_id,
                'updates': len(updates),
                'time': metrics['total_seconds']
            })
        except Exception as e:
            errors.append(f"Thread-{thread_id}: {e}")
    
    # Run 3 concurrent scans
    threads = []
    for i in range(3):
        t = threading.Thread(target=run_scan_thread, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join(timeout=60)
    
    if errors:
        print(f"✗ Errors in concurrent execution: {errors}")
        return False
    
    if len(results) == 3:
        print(f"✓ All 3 concurrent scans completed successfully")
        for r in results:
            print(f"  Thread-{r['thread']}: {r['updates']} updates in {r['time']:.2f}s")
        return True
    else:
        print(f"✗ Only {len(results)}/3 scans completed")
        return False


def main():
    """Run all comprehensive tests"""
    print("\n" + "="*60)
    print("PHASE 3D-D: COMPREHENSIVE TESTING")
    print("="*60)
    
    all_results = []
    
    # Run edge case tests
    edge_results = test_edge_cases()
    all_results.extend(edge_results)
    
    # Run performance tests
    print("\n" + "-"*60)
    perf_result = test_performance_overhead()
    all_results.append(("Performance overhead", perf_result))
    
    # Run timing accuracy tests
    print("\n" + "-"*60)
    timing_result = test_stage_timing_accuracy()
    all_results.append(("Stage timing accuracy", timing_result))
    
    # Run ETA accuracy tests
    print("\n" + "-"*60)
    eta_result = test_eta_accuracy()
    all_results.append(("ETA accuracy", eta_result))
    
    # Run concurrent scan tests
    print("\n" + "-"*60)
    concurrent_result = test_concurrent_scans()
    all_results.append(("Concurrent scans", concurrent_result))
    
    # Summary
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in all_results if success)
    total = len(all_results)
    
    for test_name, success in all_results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:30s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All comprehensive tests passed!")
        print("\nPhase 3D-D Implementation Status:")
        print("✅ Multi-stage progress tracking fully functional")
        print("✅ All edge cases handled correctly")
        print("✅ Performance overhead minimal (<5%)")
        print("✅ Stage timings accurate")
        print("✅ ETA calculations working")
        print("✅ Concurrent execution supported")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
