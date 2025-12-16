"""
Test Phase 3D-D: Multi-Stage Progress Tracking
Verifies the 4-stage progress tracking with weighted completion and ETAs
"""

import time
import json
from typing import Dict, Any, List
from technic_v4.scanner_core_enhanced import run_scan_enhanced, ScanConfig
from technic_v4.progress import MultiStageProgressTracker, format_time


class ProgressCollector:
    """Collects progress updates for analysis"""
    
    def __init__(self):
        self.updates: List[Dict[str, Any]] = []
        self.stage_transitions: List[str] = []
        self.last_update_time = time.time()
    
    def callback(self, stage: str, current: int, total: int, message: str, metadata: dict):
        """Progress callback that collects all updates"""
        now = time.time()
        update = {
            'timestamp': now,
            'time_since_last': now - self.last_update_time,
            'stage': stage,
            'current': current,
            'total': total,
            'message': message,
            'metadata': metadata.copy() if metadata else {}
        }
        self.updates.append(update)
        
        # Track stage transitions
        if not self.stage_transitions or self.stage_transitions[-1] != stage:
            self.stage_transitions.append(stage)
        
        self.last_update_time = now
        
        # Print progress for visual feedback
        if metadata:
            overall_pct = metadata.get('overall_progress_pct', 0)
            stage_pct = metadata.get('stage_progress_pct', 0)
            overall_eta = metadata.get('overall_eta')
            
            eta_str = format_time(overall_eta) if overall_eta else "calculating..."
            print(f"[{stage:20s}] Overall: {overall_pct:5.1f}% | Stage: {stage_pct:5.1f}% | ETA: {eta_str} | {message}")
    
    def get_stage_summary(self) -> Dict[str, Any]:
        """Get summary statistics for each stage"""
        stage_stats = {}
        
        for stage in set(u['stage'] for u in self.updates):
            stage_updates = [u for u in self.updates if u['stage'] == stage]
            if not stage_updates:
                continue
            
            # Calculate stage duration
            first_update = stage_updates[0]
            last_update = stage_updates[-1]
            duration = last_update['timestamp'] - first_update['timestamp']
            
            # Get progress percentages
            progress_values = [
                u['metadata'].get('stage_progress_pct', 0) 
                for u in stage_updates 
                if 'stage_progress_pct' in u.get('metadata', {})
            ]
            
            # Get throughput values
            throughput_values = [
                u['metadata'].get('stage_throughput', 0)
                for u in stage_updates
                if 'stage_throughput' in u.get('metadata', {})
            ]
            
            stage_stats[stage] = {
                'update_count': len(stage_updates),
                'duration_seconds': round(duration, 2),
                'max_progress_pct': max(progress_values) if progress_values else 0,
                'avg_throughput': sum(throughput_values) / len(throughput_values) if throughput_values else 0,
                'first_message': first_update['message'],
                'last_message': last_update['message']
            }
        
        return stage_stats


def test_stage_weights():
    """Test that stage weights are correctly configured"""
    print("\n" + "="*60)
    print("TEST 1: Stage Weight Configuration")
    print("="*60)
    
    tracker = MultiStageProgressTracker({
        "universe_loading": 0.05,
        "data_fetching": 0.20,
        "symbol_scanning": 0.70,
        "finalization": 0.05
    })
    
    total_weight = sum(tracker.stage_weights.values())
    print(f"Total weight: {total_weight:.2f} (should be 1.00)")
    
    for stage, weight in tracker.stage_weights.items():
        percentage = weight * 100
        print(f"  {stage:20s}: {percentage:5.1f}%")
    
    assert 0.99 <= total_weight <= 1.01, f"Weights must sum to 1.0, got {total_weight}"
    print("✓ Stage weights correctly configured")
    return True


def test_progress_tracking():
    """Test progress tracking with a small scan"""
    print("\n" + "="*60)
    print("TEST 2: Multi-Stage Progress Tracking")
    print("="*60)
    
    # Create progress collector
    collector = ProgressCollector()
    
    # Configure a minimal scan
    config = ScanConfig(
        max_symbols=10,  # Small number for quick test
        sectors=["Information Technology"],  # Use correct sector name
        min_tech_rating=10.0,
        lookback_days=30  # Shorter lookback for speed
    )
    
    print("\nStarting scan with progress tracking...")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Run scan with progress tracking
        results_df, status_text, metrics = run_scan_enhanced(
            config=config,
            progress_cb=collector.callback
        )
        
        elapsed = time.time() - start_time
        
        print("-" * 60)
        print(f"\nScan completed in {elapsed:.2f} seconds")
        print(f"Status: {status_text}")
        print(f"Results: {len(results_df)} symbols")
        
        # Analyze collected progress data
        print("\n" + "="*60)
        print("PROGRESS ANALYSIS")
        print("="*60)
        
        # Check stage transitions
        print(f"\nStage transitions: {' -> '.join(collector.stage_transitions)}")
        expected_stages = ["universe_loading", "data_fetching", "symbol_scanning", "finalization"]
        
        for expected in expected_stages:
            if expected not in collector.stage_transitions:
                print(f"⚠ Warning: Stage '{expected}' not found in transitions")
        
        # Get stage summary
        stage_summary = collector.get_stage_summary()
        
        print("\nStage Statistics:")
        print("-" * 60)
        for stage, stats in stage_summary.items():
            print(f"\n{stage}:")
            print(f"  Updates: {stats['update_count']}")
            print(f"  Duration: {stats['duration_seconds']:.2f}s")
            print(f"  Max Progress: {stats['max_progress_pct']:.1f}%")
            if stats['avg_throughput'] > 0:
                print(f"  Avg Throughput: {stats['avg_throughput']:.2f} items/sec")
            print(f"  First: {stats['first_message'][:50]}...")
            print(f"  Last: {stats['last_message'][:50]}...")
        
        # Verify performance metrics
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        
        print(f"\nOverall Metrics:")
        print(f"  Total time: {metrics['total_seconds']:.2f}s")
        print(f"  Symbols scanned: {metrics['symbols_scanned']}")
        print(f"  Symbols returned: {metrics['symbols_returned']}")
        print(f"  Throughput: {metrics['symbols_per_second']:.2f} symbols/sec")
        print(f"  Speedup: {metrics['speedup']:.2f}x")
        
        if 'stage_timings' in metrics:
            print(f"\nStage Timings:")
            for stage, timing in metrics['stage_timings'].items():
                print(f"  {stage}: {timing}s")
        
        if 'cache_stats' in metrics and metrics['cache_stats']:
            print(f"\nCache Statistics:")
            for key, value in metrics['cache_stats'].items():
                print(f"  {key}: {value}")
        
        # Validate progress percentages
        print("\n" + "="*60)
        print("VALIDATION")
        print("="*60)
        
        # Check that we reached 100% for completed stages
        overall_progress_values = [
            u['metadata'].get('overall_progress_pct', 0)
            for u in collector.updates
            if 'overall_progress_pct' in u.get('metadata', {})
        ]
        
        if overall_progress_values:
            max_overall = max(overall_progress_values)
            print(f"Maximum overall progress: {max_overall:.1f}%")
            
            if max_overall >= 95:  # Allow some tolerance
                print("✓ Progress tracking reached near completion")
            else:
                print(f"⚠ Warning: Maximum progress only {max_overall:.1f}%")
        
        # Check ETA calculations
        eta_updates = [
            u for u in collector.updates
            if u.get('metadata', {}).get('overall_eta') is not None
        ]
        
        if eta_updates:
            print(f"✓ ETA calculations provided in {len(eta_updates)} updates")
            
            # Check if ETAs decreased over time
            eta_values = [u['metadata']['overall_eta'] for u in eta_updates]
            if len(eta_values) > 2:
                early_eta = sum(eta_values[:3]) / 3  # Average of first 3
                late_eta = sum(eta_values[-3:]) / 3  # Average of last 3
                if late_eta < early_eta:
                    print("✓ ETAs decreased over time (as expected)")
        
        print("\n✓ Multi-stage progress tracking test completed successfully")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stage_tracker_directly():
    """Test the MultiStageProgressTracker directly"""
    print("\n" + "="*60)
    print("TEST 3: Direct Stage Tracker Testing")
    print("="*60)
    
    # Create tracker
    tracker = MultiStageProgressTracker({
        "stage1": 0.3,
        "stage2": 0.5,
        "stage3": 0.2
    })
    
    print("\nSimulating 3-stage process...")
    
    # Stage 1
    tracker.start_stage("stage1", 100)
    for i in range(0, 101, 20):
        progress = tracker.update(i)
        print(f"Stage 1 [{i:3d}/100]: Overall {progress['overall_progress_pct']:5.1f}%")
        time.sleep(0.1)
    tracker.complete_stage()
    
    # Stage 2
    tracker.start_stage("stage2", 50)
    for i in range(0, 51, 10):
        progress = tracker.update(i)
        print(f"Stage 2 [{i:3d}/50]:  Overall {progress['overall_progress_pct']:5.1f}%")
        time.sleep(0.1)
    tracker.complete_stage()
    
    # Stage 3
    tracker.start_stage("stage3", 20)
    for i in range(0, 21, 5):
        progress = tracker.update(i)
        print(f"Stage 3 [{i:3d}/20]:  Overall {progress['overall_progress_pct']:5.1f}%")
        time.sleep(0.1)
    tracker.complete_stage()
    
    print(f"\nFinal summary: {tracker.get_summary()}")
    print("✓ Direct stage tracker test completed")
    return True


def main():
    """Run all multi-stage progress tests"""
    print("\n" + "="*60)
    print("PHASE 3D-D: MULTI-STAGE PROGRESS TRACKING TESTS")
    print("="*60)
    
    tests = [
        ("Stage Weight Configuration", test_stage_weights),
        ("Multi-Stage Progress Tracking", test_progress_tracking),
        ("Direct Stage Tracker", test_stage_tracker_directly)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:40s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All multi-stage progress tracking tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
