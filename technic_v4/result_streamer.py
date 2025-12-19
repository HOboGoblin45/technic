"""
Result Streamer for Phase 3E-B: Incremental Results Streaming
Streams scan results as they complete rather than waiting for full scan
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from collections import deque
import threading
import json


@dataclass
class ScanResult:
    """Individual scan result for a symbol"""
    symbol: str
    signal: Optional[str]
    tech_rating: float
    alpha_score: float
    entry: Optional[float]
    stop: Optional[float]
    target: Optional[float]
    priority_tier: Optional[str] = None
    priority_score: Optional[float] = None
    batch_number: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'symbol': self.symbol,
            'signal': self.signal,
            'tech_rating': self.tech_rating,
            'alpha_score': self.alpha_score,
            'entry': self.entry,
            'stop': self.stop,
            'target': self.target,
            'priority_tier': self.priority_tier,
            'priority_score': self.priority_score,
            'batch_number': self.batch_number,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


@dataclass
class StreamStats:
    """Statistics for a result stream"""
    scan_id: str
    total_symbols: int
    processed: int = 0
    signals_found: int = 0
    high_value_signals: int = 0
    start_time: float = field(default_factory=time.time)
    first_result_time: Optional[float] = None
    last_result_time: Optional[float] = None
    
    def update(self, result: ScanResult):
        """Update stats with new result"""
        self.processed += 1
        
        if self.first_result_time is None:
            self.first_result_time = time.time()
        
        self.last_result_time = time.time()
        
        if result.signal:
            self.signals_found += 1
            if result.tech_rating and result.tech_rating > 70:
                self.high_value_signals += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        elapsed = time.time() - self.start_time
        time_to_first = (self.first_result_time - self.start_time) if self.first_result_time else None
        
        return {
            'scan_id': self.scan_id,
            'total_symbols': self.total_symbols,
            'processed': self.processed,
            'signals_found': self.signals_found,
            'high_value_signals': self.high_value_signals,
            'progress_pct': (self.processed / self.total_symbols * 100) if self.total_symbols > 0 else 0,
            'elapsed_time': elapsed,
            'time_to_first_result': time_to_first,
            'symbols_per_second': self.processed / elapsed if elapsed > 0 else 0
        }


class ResultQueue:
    """
    Thread-safe queue for streaming results
    
    Features:
    - Thread-safe operations
    - Batch buffering for efficiency
    - Multiple subscriber support
    - Automatic cleanup
    """
    
    def __init__(self, scan_id: str, max_buffer_size: int = 100):
        """
        Initialize result queue
        
        Args:
            scan_id: Unique scan identifier
            max_buffer_size: Maximum buffer size before forcing flush
        """
        self.scan_id = scan_id
        self.max_buffer_size = max_buffer_size
        
        # Thread-safe queue
        self._queue = deque()
        self._lock = threading.Lock()
        
        # Subscribers (callbacks to notify)
        self._subscribers: List[Callable[[ScanResult], None]] = []
        
        # State tracking
        self._is_complete = False
        self._result_count = 0
        
    def add_result(self, result: ScanResult):
        """
        Add a result to the queue
        
        Args:
            result: Scan result to add
        """
        with self._lock:
            self._queue.append(result)
            self._result_count += 1
            
            # Notify subscribers
            for subscriber in self._subscribers:
                try:
                    subscriber(result)
                except Exception as e:
                    print(f"Error notifying subscriber: {e}")
    
    def get_results(self, max_count: Optional[int] = None) -> List[ScanResult]:
        """
        Get results from queue
        
        Args:
            max_count: Maximum number of results to retrieve
            
        Returns:
            List of scan results
        """
        with self._lock:
            if max_count is None:
                results = list(self._queue)
                self._queue.clear()
            else:
                results = []
                for _ in range(min(max_count, len(self._queue))):
                    results.append(self._queue.popleft())
            
            return results
    
    def subscribe(self, callback: Callable[[ScanResult], None]):
        """
        Subscribe to result notifications
        
        Args:
            callback: Function to call when new result arrives
        """
        with self._lock:
            self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[ScanResult], None]):
        """
        Unsubscribe from notifications
        
        Args:
            callback: Callback to remove
        """
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)
    
    def mark_complete(self):
        """Mark the scan as complete"""
        with self._lock:
            self._is_complete = True
    
    def is_complete(self) -> bool:
        """Check if scan is complete"""
        with self._lock:
            return self._is_complete
    
    def get_count(self) -> int:
        """Get total result count"""
        with self._lock:
            return self._result_count
    
    def has_pending(self) -> bool:
        """Check if there are pending results"""
        with self._lock:
            return len(self._queue) > 0


class StreamManager:
    """
    Manages multiple result streams
    
    Features:
    - Multiple concurrent streams
    - Automatic cleanup
    - Statistics tracking
    - Early termination support
    """
    
    def __init__(self):
        """Initialize stream manager"""
        self._streams: Dict[str, ResultQueue] = {}
        self._stats: Dict[str, StreamStats] = {}
        self._lock = threading.Lock()
        
        # Early termination criteria
        self._termination_criteria: Dict[str, Dict[str, Any]] = {}
    
    def create_stream(
        self,
        scan_id: str,
        total_symbols: int,
        termination_criteria: Optional[Dict[str, Any]] = None
    ) -> ResultQueue:
        """
        Create a new result stream
        
        Args:
            scan_id: Unique scan identifier
            total_symbols: Total number of symbols to scan
            termination_criteria: Optional early termination rules
            
        Returns:
            Result queue for the stream
        """
        with self._lock:
            if scan_id in self._streams:
                raise ValueError(f"Stream {scan_id} already exists")
            
            queue = ResultQueue(scan_id)
            self._streams[scan_id] = queue
            self._stats[scan_id] = StreamStats(scan_id, total_symbols)
            
            if termination_criteria:
                self._termination_criteria[scan_id] = termination_criteria
            
            return queue
    
    def add_result(self, scan_id: str, result: ScanResult) -> bool:
        """
        Add result to stream
        
        Args:
            scan_id: Scan identifier
            result: Scan result
            
        Returns:
            True if should continue, False if should terminate early
        """
        with self._lock:
            if scan_id not in self._streams:
                raise ValueError(f"Stream {scan_id} not found")
            
            queue = self._streams[scan_id]
            stats = self._stats[scan_id]
            
            # Add result
            queue.add_result(result)
            stats.update(result)
            
            # Check early termination
            if scan_id in self._termination_criteria:
                if self._should_terminate(scan_id, stats):
                    queue.mark_complete()
                    return False
            
            return True
    
    def _should_terminate(self, scan_id: str, stats: StreamStats) -> bool:
        """Check if scan should terminate early"""
        criteria = self._termination_criteria.get(scan_id, {})
        
        # Check max signals
        if 'max_signals' in criteria:
            if stats.signals_found >= criteria['max_signals']:
                return True
        
        # Check max high-value signals
        if 'max_high_value' in criteria:
            if stats.high_value_signals >= criteria['max_high_value']:
                return True
        
        # Check max processed
        if 'max_processed' in criteria:
            if stats.processed >= criteria['max_processed']:
                return True
        
        # Check timeout
        if 'timeout_seconds' in criteria:
            elapsed = time.time() - stats.start_time
            if elapsed >= criteria['timeout_seconds']:
                return True
        
        return False
    
    def get_stream(self, scan_id: str) -> Optional[ResultQueue]:
        """Get stream by ID"""
        with self._lock:
            return self._streams.get(scan_id)
    
    def get_stats(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Get stream statistics"""
        with self._lock:
            stats = self._stats.get(scan_id)
            return stats.to_dict() if stats else None
    
    def complete_stream(self, scan_id: str):
        """Mark stream as complete"""
        with self._lock:
            if scan_id in self._streams:
                self._streams[scan_id].mark_complete()
    
    def cleanup_stream(self, scan_id: str):
        """Remove stream and free resources"""
        with self._lock:
            if scan_id in self._streams:
                del self._streams[scan_id]
            if scan_id in self._stats:
                del self._stats[scan_id]
            if scan_id in self._termination_criteria:
                del self._termination_criteria[scan_id]
    
    def get_active_streams(self) -> List[str]:
        """Get list of active stream IDs"""
        with self._lock:
            return list(self._streams.keys())


# Global stream manager instance
_stream_manager = StreamManager()


def get_stream_manager() -> StreamManager:
    """Get the global stream manager instance"""
    return _stream_manager


# Example usage
if __name__ == "__main__":
    print("Result Streamer Test")
    print("=" * 60)
    
    # Create stream manager
    manager = StreamManager()
    
    # Create a stream
    scan_id = "test_scan_001"
    total_symbols = 10
    
    # Set early termination: stop after 3 signals
    termination_criteria = {
        'max_signals': 3,
        'timeout_seconds': 30
    }
    
    queue = manager.create_stream(scan_id, total_symbols, termination_criteria)
    
    print(f"Created stream: {scan_id}")
    print(f"Total symbols: {total_symbols}")
    print(f"Termination criteria: {termination_criteria}")
    
    # Subscribe to results
    def on_result(result: ScanResult):
        print(f"  → Result received: {result.symbol} (signal: {result.signal})")
    
    queue.subscribe(on_result)
    
    # Simulate adding results
    print("\nSimulating scan results...")
    test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'JPM', 'V', 'JNJ']
    
    for i, symbol in enumerate(test_symbols):
        # Simulate some symbols having signals
        has_signal = (i % 3 == 0)
        
        result = ScanResult(
            symbol=symbol,
            signal='BUY' if has_signal else None,
            tech_rating=85.0 if has_signal else 45.0,
            alpha_score=0.8 if has_signal else 0.3,
            entry=100.0 if has_signal else None,
            stop=95.0 if has_signal else None,
            target=110.0 if has_signal else None,
            priority_tier='high' if i < 5 else 'medium',
            batch_number=(i // 5) + 1
        )
        
        # Add result
        should_continue = manager.add_result(scan_id, result)
        
        if not should_continue:
            print(f"\n✓ Early termination triggered after {i + 1} symbols")
            break
        
        time.sleep(0.1)  # Simulate processing time
    
    # Get final stats
    print("\n" + "=" * 60)
    print("Final Statistics:")
    stats = manager.get_stats(scan_id)
    if stats:
        print(f"  Processed: {stats['processed']}/{stats['total_symbols']}")
        print(f"  Signals found: {stats['signals_found']}")
        print(f"  High-value signals: {stats['high_value_signals']}")
        print(f"  Progress: {stats['progress_pct']:.1f}%")
        print(f"  Time to first result: {stats['time_to_first_result']:.3f}s")
        print(f"  Throughput: {stats['symbols_per_second']:.2f} symbols/sec")
    
    # Cleanup
    manager.cleanup_stream(scan_id)
    print("\n✓ Stream cleaned up")
