"""
Progress tracking utilities with time estimation
Provides real-time progress updates with ETA calculations
"""
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class ProgressTracker:
    """
    Track progress with time estimation and throughput calculation
    
    Example:
        >>> tracker = ProgressTracker(total=100)
        >>> progress = tracker.update(25)
        >>> print(f"Progress: {progress['progress_pct']}%")
        >>> print(f"ETA: {progress['estimated_remaining']}s")
    """
    total: int
    start_time: float = field(default_factory=time.time)
    current: int = 0
    stage: str = "processing"
    
    def update(self, current: int, stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Update progress and calculate time estimates
        
        Args:
            current: Current progress count
            stage: Optional stage name (e.g., "scanning", "filtering")
        
        Returns:
            Dictionary with progress information including:
            - current: Current count
            - total: Total count
            - progress_pct: Percentage complete
            - elapsed_time: Time elapsed in seconds
            - estimated_remaining: Estimated time remaining in seconds
            - estimated_total: Estimated total time in seconds
            - throughput: Items per second
            - stage: Current stage name
        """
        self.current = current
        if stage:
            self.stage = stage
            
        elapsed = time.time() - self.start_time
        
        # Handle initial state
        if current == 0 or elapsed == 0:
            return {
                "current": 0,
                "total": self.total,
                "progress_pct": 0.0,
                "elapsed_time": 0.0,
                "estimated_remaining": None,
                "estimated_total": None,
                "throughput": 0.0,
                "stage": self.stage
            }
        
        # Calculate metrics
        progress_pct = (current / self.total * 100) if self.total > 0 else 0
        throughput = current / elapsed if elapsed > 0 else 0
        
        # Estimate remaining time
        if throughput > 0 and current < self.total:
            remaining_items = self.total - current
            estimated_remaining = remaining_items / throughput
            estimated_total = elapsed + estimated_remaining
        else:
            estimated_remaining = None
            estimated_total = elapsed if current >= self.total else None
        
        return {
            "current": current,
            "total": self.total,
            "progress_pct": round(progress_pct, 1),
            "elapsed_time": round(elapsed, 1),
            "estimated_remaining": round(estimated_remaining, 1) if estimated_remaining else None,
            "estimated_total": round(estimated_total, 1) if estimated_total else None,
            "throughput": round(throughput, 2),
            "stage": self.stage
        }
    
    def reset(self, total: Optional[int] = None):
        """
        Reset the progress tracker
        
        Args:
            total: New total count (optional, keeps current if not provided)
        """
        if total is not None:
            self.total = total
        self.current = 0
        self.start_time = time.time()
    
    def is_complete(self) -> bool:
        """Check if progress is complete"""
        return self.current >= self.total
    
    def get_summary(self) -> str:
        """
        Get a human-readable progress summary
        
        Returns:
            String like "45/100 (45.0%) - ETA: 15.2s"
        """
        progress = self.update(self.current)
        
        summary = f"{progress['current']}/{progress['total']} ({progress['progress_pct']}%)"
        
        if progress['estimated_remaining']:
            summary += f" - ETA: {progress['estimated_remaining']}s"
        elif self.is_complete():
            summary += f" - Complete in {progress['elapsed_time']}s"
        
        return summary


class MultiStageProgressTracker:
    """
    Track progress across multiple stages with weighted completion
    
    Example:
        >>> tracker = MultiStageProgressTracker({
        ...     "prefetch": 0.2,  # 20% of total
        ...     "scan": 0.7,      # 70% of total
        ...     "filter": 0.1     # 10% of total
        ... })
        >>> tracker.start_stage("prefetch", total=100)
        >>> progress = tracker.update(50)
        >>> print(f"Overall: {progress['overall_progress_pct']}%")
    """
    
    def __init__(self, stage_weights: Dict[str, float]):
        """
        Initialize multi-stage tracker
        
        Args:
            stage_weights: Dictionary mapping stage names to their weight (0-1)
                          Weights should sum to 1.0
        """
        self.stage_weights = stage_weights
        self.stages: Dict[str, ProgressTracker] = {}
        self.current_stage: Optional[str] = None
        self.completed_stages: set = set()
        self.overall_start_time = time.time()
        
        # Validate weights
        total_weight = sum(stage_weights.values())
        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Stage weights must sum to 1.0, got {total_weight}")
    
    def start_stage(self, stage_name: str, total: int):
        """
        Start a new stage
        
        Args:
            stage_name: Name of the stage
            total: Total items in this stage
        """
        if stage_name not in self.stage_weights:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        self.current_stage = stage_name
        self.stages[stage_name] = ProgressTracker(total=total, stage=stage_name)
    
    def update(self, current: int) -> Dict[str, Any]:
        """
        Update current stage progress
        
        Args:
            current: Current progress in the active stage
        
        Returns:
            Dictionary with both stage and overall progress
        """
        if not self.current_stage:
            raise ValueError("No active stage. Call start_stage() first.")
        
        # Update current stage
        stage_progress = self.stages[self.current_stage].update(current)
        
        # Calculate overall progress
        overall_progress = 0.0
        for stage_name, weight in self.stage_weights.items():
            if stage_name in self.completed_stages:
                overall_progress += weight * 100
            elif stage_name == self.current_stage:
                overall_progress += weight * stage_progress['progress_pct']
        
        overall_elapsed = time.time() - self.overall_start_time
        
        # Estimate overall remaining time
        if overall_progress > 0:
            estimated_total = (overall_elapsed / overall_progress) * 100
            estimated_remaining = estimated_total - overall_elapsed
        else:
            estimated_remaining = None
            estimated_total = None
        
        return {
            "stage": self.current_stage,
            "stage_progress": stage_progress,
            "overall_progress_pct": round(overall_progress, 1),
            "overall_elapsed_time": round(overall_elapsed, 1),
            "overall_estimated_remaining": round(estimated_remaining, 1) if estimated_remaining else None,
            "overall_estimated_total": round(estimated_total, 1) if estimated_total else None,
            "completed_stages": list(self.completed_stages)
        }
    
    def complete_stage(self):
        """Mark the current stage as complete"""
        if self.current_stage:
            self.completed_stages.add(self.current_stage)
            self.current_stage = None
    
    def is_complete(self) -> bool:
        """Check if all stages are complete"""
        return len(self.completed_stages) == len(self.stage_weights)
    
    def get_summary(self) -> str:
        """
        Get a human-readable summary of overall progress
        
        Returns:
            String like "Overall: 65.0% (Stage: scan 50/100)"
        """
        if not self.current_stage:
            if self.is_complete():
                elapsed = time.time() - self.overall_start_time
                return f"Complete - Total time: {elapsed:.1f}s"
            return "No active stage"
        
        progress = self.update(self.stages[self.current_stage].current)
        stage_info = progress['stage_progress']
        
        return (
            f"Overall: {progress['overall_progress_pct']}% "
            f"(Stage: {self.current_stage} {stage_info['current']}/{stage_info['total']})"
        )


def format_time(seconds: Optional[float]) -> str:
    """
    Format seconds into human-readable time
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted string like "1m 30s" or "45s"
    
    Example:
        >>> format_time(90)
        '1m 30s'
        >>> format_time(45)
        '45s'
        >>> format_time(None)
        'N/A'
    """
    if seconds is None:
        return "N/A"
    
    if seconds < 60:
        return f"{int(seconds)}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    
    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"
    
    hours = int(minutes // 60)
    remaining_minutes = int(minutes % 60)
    return f"{hours}h {remaining_minutes}m"


def format_throughput(items_per_second: float, unit: str = "items") -> str:
    """
    Format throughput into human-readable string
    
    Args:
        items_per_second: Throughput rate
        unit: Unit name (e.g., "items", "symbols", "requests")
    
    Returns:
        Formatted string like "3.5 symbols/sec"
    
    Example:
        >>> format_throughput(3.5, "symbols")
        '3.5 symbols/sec'
    """
    return f"{items_per_second:.1f} {unit}/sec"
