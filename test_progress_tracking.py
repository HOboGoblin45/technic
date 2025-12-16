"""
Test progress tracking system for Phase 3D-B
"""
import time
import pytest
from technic_v4.progress import (
    ProgressTracker,
    MultiStageProgressTracker,
    format_time,
    format_throughput
)


def test_progress_tracker_initialization():
    """Test basic progress tracker initialization"""
    tracker = ProgressTracker(total=100)
    
    assert tracker.total == 100
    assert tracker.current == 0
    assert tracker.stage == "processing"
    print("✅ Progress tracker initialization test passed")


def test_progress_tracker_update():
    """Test progress updates and calculations"""
    tracker = ProgressTracker(total=100)
    
    # Simulate some progress
    time.sleep(0.1)
    progress = tracker.update(25)
    
    assert progress["current"] == 25
    assert progress["total"] == 100
    assert progress["progress_pct"] == 25.0
    assert progress["elapsed_time"] > 0
    assert progress["throughput"] > 0
    assert progress["estimated_remaining"] is not None
    assert progress["estimated_total"] is not None
    print("✅ Progress tracker update test passed")


def test_progress_tracker_zero_state():
    """Test progress tracker at zero progress"""
    tracker = ProgressTracker(total=100)
    progress = tracker.update(0)
    
    assert progress["current"] == 0
    assert progress["progress_pct"] == 0.0
    assert progress["elapsed_time"] == 0.0
    assert progress["estimated_remaining"] is None
    assert progress["throughput"] == 0.0
    print("✅ Progress tracker zero state test passed")


def test_progress_tracker_completion():
    """Test progress tracker at 100% completion"""
    tracker = ProgressTracker(total=100)
    
    time.sleep(0.1)
    progress = tracker.update(100)
    
    assert progress["current"] == 100
    assert progress["progress_pct"] == 100.0
    assert tracker.is_complete() == True
    print("✅ Progress tracker completion test passed")


def test_progress_tracker_throughput():
    """Test throughput calculation"""
    tracker = ProgressTracker(total=100)
    
    # Simulate processing 50 items in 1 second
    time.sleep(1.0)
    progress = tracker.update(50)
    
    # Throughput should be approximately 50 items/sec
    assert 45 <= progress["throughput"] <= 55  # Allow some variance
    print(f"✅ Progress tracker throughput test passed (throughput: {progress['throughput']:.2f} items/sec)")


def test_progress_tracker_eta():
    """Test ETA calculation"""
    tracker = ProgressTracker(total=100)
    
    # Simulate processing 25 items in 1 second
    time.sleep(1.0)
    progress = tracker.update(25)
    
    # Should estimate ~3 seconds remaining (75 items at 25 items/sec)
    assert progress["estimated_remaining"] is not None
    assert 2.5 <= progress["estimated_remaining"] <= 3.5  # Allow some variance
    print(f"✅ Progress tracker ETA test passed (ETA: {progress['estimated_remaining']:.1f}s)")


def test_progress_tracker_reset():
    """Test progress tracker reset"""
    tracker = ProgressTracker(total=100)
    tracker.update(50)
    
    # Reset with new total
    tracker.reset(total=200)
    
    assert tracker.total == 200
    assert tracker.current == 0
    assert not tracker.is_complete()
    print("✅ Progress tracker reset test passed")


def test_progress_tracker_stage():
    """Test progress tracker with stage names"""
    tracker = ProgressTracker(total=100, stage="scanning")
    
    progress = tracker.update(25, stage="filtering")
    
    assert progress["stage"] == "filtering"
    assert tracker.stage == "filtering"
    print("✅ Progress tracker stage test passed")


def test_progress_tracker_summary():
    """Test progress summary generation"""
    tracker = ProgressTracker(total=100)
    
    time.sleep(0.1)
    tracker.update(45)
    summary = tracker.get_summary()
    
    assert "45/100" in summary
    assert "45.0%" in summary
    assert "ETA:" in summary
    print(f"✅ Progress tracker summary test passed: {summary}")


def test_multi_stage_tracker_initialization():
    """Test multi-stage tracker initialization"""
    stage_weights = {
        "prefetch": 0.2,
        "scan": 0.7,
        "filter": 0.1
    }
    
    tracker = MultiStageProgressTracker(stage_weights)
    
    assert len(tracker.stage_weights) == 3
    assert tracker.current_stage is None
    assert len(tracker.completed_stages) == 0
    print("✅ Multi-stage tracker initialization test passed")


def test_multi_stage_tracker_invalid_weights():
    """Test multi-stage tracker with invalid weights"""
    stage_weights = {
        "stage1": 0.5,
        "stage2": 0.3  # Sum is 0.8, not 1.0
    }
    
    with pytest.raises(ValueError):
        MultiStageProgressTracker(stage_weights)
    
    print("✅ Multi-stage tracker invalid weights test passed")


def test_multi_stage_tracker_single_stage():
    """Test multi-stage tracker with single stage"""
    stage_weights = {
        "prefetch": 0.3,
        "scan": 0.7
    }
    
    tracker = MultiStageProgressTracker(stage_weights)
    tracker.start_stage("prefetch", total=100)
    
    time.sleep(0.1)
    progress = tracker.update(50)
    
    assert progress["stage"] == "prefetch"
    assert progress["stage_progress"]["current"] == 50
    assert progress["stage_progress"]["progress_pct"] == 50.0
    # Overall progress should be 50% of 30% = 15%
    assert 14 <= progress["overall_progress_pct"] <= 16
    print(f"✅ Multi-stage tracker single stage test passed (overall: {progress['overall_progress_pct']}%)")


def test_multi_stage_tracker_complete_stage():
    """Test completing a stage in multi-stage tracker"""
    stage_weights = {
        "stage1": 0.5,
        "stage2": 0.5
    }
    
    tracker = MultiStageProgressTracker(stage_weights)
    
    # Complete first stage
    tracker.start_stage("stage1", total=100)
    time.sleep(0.1)
    tracker.update(100)
    tracker.complete_stage()
    
    assert "stage1" in tracker.completed_stages
    assert tracker.current_stage is None
    
    # Start second stage
    tracker.start_stage("stage2", total=50)
    time.sleep(0.1)
    progress = tracker.update(25)
    
    # Overall should be 50% (stage1) + 25% (50% of stage2) = 75%
    assert 74 <= progress["overall_progress_pct"] <= 76
    print(f"✅ Multi-stage tracker complete stage test passed (overall: {progress['overall_progress_pct']}%)")


def test_multi_stage_tracker_all_complete():
    """Test multi-stage tracker when all stages complete"""
    stage_weights = {
        "stage1": 0.5,
        "stage2": 0.5
    }
    
    tracker = MultiStageProgressTracker(stage_weights)
    
    # Complete both stages
    tracker.start_stage("stage1", total=100)
    tracker.update(100)
    tracker.complete_stage()
    
    tracker.start_stage("stage2", total=100)
    tracker.update(100)
    tracker.complete_stage()
    
    assert tracker.is_complete() == True
    assert len(tracker.completed_stages) == 2
    print("✅ Multi-stage tracker all complete test passed")


def test_multi_stage_tracker_summary():
    """Test multi-stage tracker summary"""
    stage_weights = {
        "prefetch": 0.2,
        "scan": 0.8
    }
    
    tracker = MultiStageProgressTracker(stage_weights)
    tracker.start_stage("scan", total=100)
    time.sleep(0.1)
    tracker.update(50)
    
    summary = tracker.get_summary()
    
    assert "Overall:" in summary
    assert "scan" in summary
    assert "50/100" in summary
    print(f"✅ Multi-stage tracker summary test passed: {summary}")


def test_format_time_seconds():
    """Test time formatting for seconds"""
    assert format_time(45) == "45s"
    assert format_time(30.5) == "30s"
    print("✅ Format time seconds test passed")


def test_format_time_minutes():
    """Test time formatting for minutes"""
    assert format_time(90) == "1m 30s"
    assert format_time(125) == "2m 5s"
    print("✅ Format time minutes test passed")


def test_format_time_hours():
    """Test time formatting for hours"""
    assert format_time(3665) == "1h 1m"
    assert format_time(7200) == "2h 0m"
    print("✅ Format time hours test passed")


def test_format_time_none():
    """Test time formatting for None"""
    assert format_time(None) == "N/A"
    print("✅ Format time None test passed")


def test_format_throughput():
    """Test throughput formatting"""
    result = format_throughput(3.5, "symbols")
    assert "3.5" in result
    assert "symbols/sec" in result
    print(f"✅ Format throughput test passed: {result}")


def test_progress_tracker_edge_cases():
    """Test edge cases for progress tracker"""
    # Zero total
    tracker = ProgressTracker(total=0)
    progress = tracker.update(0)
    assert progress["progress_pct"] == 0.0
    
    # Progress exceeds total
    tracker = ProgressTracker(total=100)
    time.sleep(0.1)
    progress = tracker.update(150)
    assert progress["current"] == 150
    assert progress["progress_pct"] == 150.0
    
    print("✅ Progress tracker edge cases test passed")


if __name__ == "__main__":
    print("="*80)
    print("TESTING PROGRESS TRACKING SYSTEM")
    print("="*80)
    print()
    
    # Run all tests
    test_progress_tracker_initialization()
    test_progress_tracker_update()
    test_progress_tracker_zero_state()
    test_progress_tracker_completion()
    test_progress_tracker_throughput()
    test_progress_tracker_eta()
    test_progress_tracker_reset()
    test_progress_tracker_stage()
    test_progress_tracker_summary()
    test_multi_stage_tracker_initialization()
    test_multi_stage_tracker_invalid_weights()
    test_multi_stage_tracker_single_stage()
    test_multi_stage_tracker_complete_stage()
    test_multi_stage_tracker_all_complete()
    test_multi_stage_tracker_summary()
    test_format_time_seconds()
    test_format_time_minutes()
    test_format_time_hours()
    test_format_time_none()
    test_format_throughput()
    test_progress_tracker_edge_cases()
    
    print()
    print("="*80)
    print("✅ ALL PROGRESS TRACKING TESTS PASSED!")
    print("="*80)
