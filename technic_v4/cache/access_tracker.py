"""
Access Pattern Tracker for Smart Cache Warming
Tracks symbol access patterns to enable predictive caching
Path 1 Task 6: Smart Cache Warming
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import threading


@dataclass
class AccessRecord:
    """Record of a symbol access"""
    symbol: str
    timestamp: float
    hour: int
    day_of_week: int
    context: Dict[str, any] = None


class AccessPatternTracker:
    """
    Track symbol access patterns for intelligent cache warming
    
    Features:
    - Symbol access frequency tracking
    - Time-of-day pattern analysis
    - User behavior prediction
    - Persistent storage
    
    Example:
        >>> tracker = AccessPatternTracker()
        >>> tracker.track_access("AAPL")
        >>> popular = tracker.get_popular_symbols(limit=10)
        >>> print(popular)
        ['AAPL', 'MSFT', 'GOOGL', ...]
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize access pattern tracker
        
        Args:
            storage_path: Path to store access data (default: logs/access_patterns.json)
        """
        self.storage_path = storage_path or Path("logs/access_patterns.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory tracking
        self.access_counts = Counter()  # symbol -> count
        self.access_times = defaultdict(list)  # symbol -> [timestamps]
        self.hourly_patterns = defaultdict(lambda: defaultdict(int))  # hour -> {symbol: count}
        self.daily_patterns = defaultdict(lambda: defaultdict(int))  # day -> {symbol: count}
        self.recent_accesses = []  # Recent access records
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Load existing data
        self._load_data()
    
    def track_access(
        self,
        symbol: str,
        timestamp: Optional[float] = None,
        context: Optional[Dict] = None
    ):
        """
        Track a symbol access
        
        Args:
            symbol: Stock symbol
            timestamp: Access timestamp (default: now)
            context: Additional context (sector, user_id, etc.)
        """
        if timestamp is None:
            timestamp = time.time()
        
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        day_of_week = dt.weekday()
        
        with self.lock:
            # Update counters
            self.access_counts[symbol] += 1
            self.access_times[symbol].append(timestamp)
            self.hourly_patterns[hour][symbol] += 1
            self.daily_patterns[day_of_week][symbol] += 1
            
            # Store recent access
            record = AccessRecord(
                symbol=symbol,
                timestamp=timestamp,
                hour=hour,
                day_of_week=day_of_week,
                context=context
            )
            self.recent_accesses.append(record)
            
            # Keep only last 1000 records
            if len(self.recent_accesses) > 1000:
                self.recent_accesses = self.recent_accesses[-1000:]
    
    def get_popular_symbols(self, limit: int = 100) -> List[str]:
        """
        Get most frequently accessed symbols
        
        Args:
            limit: Maximum number of symbols to return
        
        Returns:
            List of symbols ordered by access frequency
        """
        with self.lock:
            return [symbol for symbol, _ in self.access_counts.most_common(limit)]
    
    def get_popular_by_time(self, hour: int, limit: int = 50) -> List[str]:
        """
        Get popular symbols for a specific hour
        
        Args:
            hour: Hour of day (0-23)
            limit: Maximum number of symbols
        
        Returns:
            List of symbols popular at that hour
        """
        with self.lock:
            if hour not in self.hourly_patterns:
                return []
            
            counts = self.hourly_patterns[hour]
            sorted_symbols = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            return [symbol for symbol, _ in sorted_symbols[:limit]]
    
    def get_trending_symbols(self, window_hours: int = 24, limit: int = 50) -> List[str]:
        """
        Get symbols trending in recent time window
        
        Args:
            window_hours: Time window in hours
            limit: Maximum number of symbols
        
        Returns:
            List of trending symbols
        """
        cutoff = time.time() - (window_hours * 3600)
        
        with self.lock:
            recent_counts = Counter()
            
            for symbol, timestamps in self.access_times.items():
                recent = [ts for ts in timestamps if ts > cutoff]
                if recent:
                    recent_counts[symbol] = len(recent)
            
            return [symbol for symbol, _ in recent_counts.most_common(limit)]
    
    def predict_next_symbols(
        self,
        current_symbols: List[str],
        limit: int = 20
    ) -> List[str]:
        """
        Predict likely next symbols based on patterns
        
        Args:
            current_symbols: Symbols currently being accessed
            limit: Maximum predictions
        
        Returns:
            List of predicted symbols
        """
        # Simple co-occurrence based prediction
        predictions = Counter()
        
        with self.lock:
            # Look at recent access sequences
            for i in range(len(self.recent_accesses) - 1):
                current = self.recent_accesses[i].symbol
                next_sym = self.recent_accesses[i + 1].symbol
                
                if current in current_symbols:
                    predictions[next_sym] += 1
        
        # Remove symbols already in current list
        for sym in current_symbols:
            predictions.pop(sym, None)
        
        return [symbol for symbol, _ in predictions.most_common(limit)]
    
    def get_access_stats(self) -> Dict:
        """
        Get comprehensive access statistics
        
        Returns:
            Dictionary with access statistics
        """
        with self.lock:
            total_accesses = sum(self.access_counts.values())
            unique_symbols = len(self.access_counts)
            
            # Calculate time range
            all_times = []
            for times in self.access_times.values():
                all_times.extend(times)
            
            time_range = None
            if all_times:
                time_range = {
                    'start': datetime.fromtimestamp(min(all_times)).isoformat(),
                    'end': datetime.fromtimestamp(max(all_times)).isoformat(),
                    'duration_hours': (max(all_times) - min(all_times)) / 3600
                }
            
            return {
                'total_accesses': total_accesses,
                'unique_symbols': unique_symbols,
                'avg_accesses_per_symbol': total_accesses / unique_symbols if unique_symbols > 0 else 0,
                'time_range': time_range,
                'top_10_symbols': self.get_popular_symbols(limit=10),
                'recent_accesses': len(self.recent_accesses)
            }
    
    def get_hourly_distribution(self) -> Dict[int, int]:
        """
        Get access distribution by hour
        
        Returns:
            Dictionary mapping hour to access count
        """
        with self.lock:
            distribution = {}
            for hour in range(24):
                if hour in self.hourly_patterns:
                    distribution[hour] = sum(self.hourly_patterns[hour].values())
                else:
                    distribution[hour] = 0
            return distribution
    
    def save_data(self):
        """Save access data to disk"""
        with self.lock:
            data = {
                'access_counts': dict(self.access_counts),
                'hourly_patterns': {
                    str(hour): dict(symbols)
                    for hour, symbols in self.hourly_patterns.items()
                },
                'daily_patterns': {
                    str(day): dict(symbols)
                    for day, symbols in self.daily_patterns.items()
                },
                'recent_accesses': [
                    asdict(record) for record in self.recent_accesses[-1000:]
                ],
                'last_updated': time.time()
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    def _load_data(self):
        """Load access data from disk"""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            # Restore counters
            self.access_counts = Counter(data.get('access_counts', {}))
            
            # Restore hourly patterns
            for hour_str, symbols in data.get('hourly_patterns', {}).items():
                hour = int(hour_str)
                self.hourly_patterns[hour] = defaultdict(int, symbols)
            
            # Restore daily patterns
            for day_str, symbols in data.get('daily_patterns', {}).items():
                day = int(day_str)
                self.daily_patterns[day] = defaultdict(int, symbols)
            
            # Restore recent accesses
            for record_dict in data.get('recent_accesses', []):
                record = AccessRecord(**record_dict)
                self.recent_accesses.append(record)
            
            # Rebuild access_times from recent accesses
            for record in self.recent_accesses:
                self.access_times[record.symbol].append(record.timestamp)
        
        except Exception as e:
            print(f"Warning: Failed to load access data: {e}")
    
    def clear_old_data(self, days: int = 30):
        """
        Clear access data older than specified days
        
        Args:
            days: Number of days to keep
        """
        cutoff = time.time() - (days * 24 * 3600)
        
        with self.lock:
            # Clear old timestamps
            for symbol in list(self.access_times.keys()):
                self.access_times[symbol] = [
                    ts for ts in self.access_times[symbol]
                    if ts > cutoff
                ]
                
                # Remove symbol if no recent accesses
                if not self.access_times[symbol]:
                    del self.access_times[symbol]
                    del self.access_counts[symbol]
            
            # Clear old recent accesses
            self.recent_accesses = [
                record for record in self.recent_accesses
                if record.timestamp > cutoff
            ]


# Global tracker instance
_global_tracker = None


def get_tracker() -> AccessPatternTracker:
    """Get global access pattern tracker instance"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = AccessPatternTracker()
    return _global_tracker


def track_symbol_access(symbol: str, context: Optional[Dict] = None):
    """
    Convenience function to track symbol access
    
    Args:
        symbol: Stock symbol
        context: Additional context
    """
    tracker = get_tracker()
    tracker.track_access(symbol, context=context)


if __name__ == "__main__":
    # Example usage
    tracker = AccessPatternTracker()
    
    # Simulate some accesses
    symbols = ["AAPL", "MSFT", "GOOGL", "AAPL", "TSLA", "AAPL", "MSFT"]
    for symbol in symbols:
        tracker.track_access(symbol)
        time.sleep(0.1)
    
    # Get statistics
    print("Access Statistics:")
    print(json.dumps(tracker.get_access_stats(), indent=2))
    
    print("\nTop 5 Popular Symbols:")
    print(tracker.get_popular_symbols(limit=5))
    
    print("\nTrending Symbols (last 24h):")
    print(tracker.get_trending_symbols(window_hours=24, limit=5))
    
    print("\nPredicted Next Symbols:")
    print(tracker.predict_next_symbols(["AAPL"], limit=3))
    
    # Save data
    tracker.save_data()
    print(f"\nData saved to: {tracker.storage_path}")
