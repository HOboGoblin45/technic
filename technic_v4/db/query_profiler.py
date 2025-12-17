"""
Query Performance Profiler
Measures and analyzes database query performance
"""

import time
import functools
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path


@dataclass
class QueryProfile:
    """Profile data for a single query execution"""
    query_name: str
    execution_time: float
    timestamp: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)
    result_count: Optional[int] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'query_name': self.query_name,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat(),
            'parameters': self.parameters,
            'result_count': self.result_count,
            'error': self.error
        }


class QueryProfiler:
    """
    Profiles database query performance
    
    Features:
    - Automatic timing of queries
    - Statistical analysis (min, max, avg, p95, p99)
    - Slow query detection
    - Performance trend tracking
    - JSON export for analysis
    
    Example:
        >>> profiler = QueryProfiler()
        >>> 
        >>> @profiler.profile("get_symbol_data")
        >>> def get_symbol_data(symbol):
        >>>     return fetch_from_db(symbol)
        >>> 
        >>> stats = profiler.get_stats()
        >>> print(f"Average time: {stats['get_symbol_data']['avg_time']:.2f}ms")
    """
    
    def __init__(self, slow_query_threshold: float = 1000.0):
        """
        Initialize profiler
        
        Args:
            slow_query_threshold: Threshold in ms for slow query detection
        """
        self.profiles: Dict[str, List[QueryProfile]] = {}
        self.slow_query_threshold = slow_query_threshold
        self.enabled = True
    
    def profile(self, query_name: str):
        """
        Decorator to profile a query function
        
        Args:
            query_name: Name to identify the query
        
        Returns:
            Decorated function that profiles execution
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                error = None
                result = None
                result_count = None
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Try to get result count
                    if hasattr(result, '__len__'):
                        try:
                            result_count = len(result)
                        except:
                            pass
                    
                    return result
                    
                except Exception as e:
                    error = str(e)
                    raise
                    
                finally:
                    execution_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    profile = QueryProfile(
                        query_name=query_name,
                        execution_time=execution_time,
                        timestamp=datetime.now(),
                        parameters={'args': str(args)[:100], 'kwargs': str(kwargs)[:100]},
                        result_count=result_count,
                        error=error
                    )
                    
                    if query_name not in self.profiles:
                        self.profiles[query_name] = []
                    
                    self.profiles[query_name].append(profile)
                    
                    # Log slow queries
                    if execution_time > self.slow_query_threshold:
                        print(f"[SLOW QUERY] {query_name}: {execution_time:.2f}ms")
            
            return wrapper
        return decorator
    
    def get_stats(self, query_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistical analysis of query performance
        
        Args:
            query_name: Specific query to analyze, or None for all queries
        
        Returns:
            Dictionary with statistics for each query
        """
        if query_name:
            queries_to_analyze = {query_name: self.profiles.get(query_name, [])}
        else:
            queries_to_analyze = self.profiles
        
        stats = {}
        
        for name, profiles in queries_to_analyze.items():
            if not profiles:
                continue
            
            times = [p.execution_time for p in profiles if p.error is None]
            
            if not times:
                stats[name] = {
                    'count': len(profiles),
                    'errors': len([p for p in profiles if p.error]),
                    'avg_time': 0,
                    'min_time': 0,
                    'max_time': 0,
                    'p95_time': 0,
                    'p99_time': 0,
                    'slow_queries': 0
                }
                continue
            
            times_sorted = sorted(times)
            count = len(times)
            
            stats[name] = {
                'count': count,
                'errors': len([p for p in profiles if p.error]),
                'avg_time': sum(times) / count,
                'min_time': times_sorted[0],
                'max_time': times_sorted[-1],
                'p95_time': times_sorted[int(count * 0.95)] if count > 0 else 0,
                'p99_time': times_sorted[int(count * 0.99)] if count > 0 else 0,
                'slow_queries': len([t for t in times if t > self.slow_query_threshold]),
                'total_time': sum(times),
                'avg_result_count': sum(p.result_count for p in profiles if p.result_count) / count if count > 0 else 0
            }
        
        return stats
    
    def get_slow_queries(self, threshold: Optional[float] = None) -> List[QueryProfile]:
        """
        Get all slow queries above threshold
        
        Args:
            threshold: Custom threshold in ms, or use default
        
        Returns:
            List of slow query profiles
        """
        threshold = threshold or self.slow_query_threshold
        slow_queries = []
        
        for profiles in self.profiles.values():
            slow_queries.extend([
                p for p in profiles 
                if p.execution_time > threshold and p.error is None
            ])
        
        return sorted(slow_queries, key=lambda p: p.execution_time, reverse=True)
    
    def get_top_queries(self, n: int = 10, by: str = 'avg_time') -> List[tuple]:
        """
        Get top N slowest queries
        
        Args:
            n: Number of queries to return
            by: Metric to sort by ('avg_time', 'max_time', 'total_time', 'count')
        
        Returns:
            List of (query_name, stats) tuples
        """
        stats = self.get_stats()
        
        if by not in ['avg_time', 'max_time', 'total_time', 'count']:
            by = 'avg_time'
        
        sorted_queries = sorted(
            stats.items(),
            key=lambda x: x[1].get(by, 0),
            reverse=True
        )
        
        return sorted_queries[:n]
    
    def export_to_json(self, filepath: str):
        """
        Export all profiles to JSON file
        
        Args:
            filepath: Path to save JSON file
        """
        data = {
            'exported_at': datetime.now().isoformat(),
            'slow_query_threshold': self.slow_query_threshold,
            'stats': self.get_stats(),
            'profiles': {
                name: [p.to_dict() for p in profiles]
                for name, profiles in self.profiles.items()
            }
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[PROFILER] Exported {len(self.profiles)} query profiles to {filepath}")
    
    def reset(self):
        """Clear all collected profiles"""
        self.profiles.clear()
        print("[PROFILER] Reset all profiles")
    
    def enable(self):
        """Enable profiling"""
        self.enabled = True
    
    def disable(self):
        """Disable profiling"""
        self.enabled = False
    
    def print_summary(self):
        """Print a summary of query performance"""
        stats = self.get_stats()
        
        if not stats:
            print("[PROFILER] No queries profiled yet")
            return
        
        print("\n" + "="*80)
        print("QUERY PERFORMANCE SUMMARY")
        print("="*80)
        
        total_queries = sum(s['count'] for s in stats.values())
        total_time = sum(s['total_time'] for s in stats.values())
        total_slow = sum(s['slow_queries'] for s in stats.values())
        
        print(f"\nOverall Statistics:")
        print(f"  Total Queries: {total_queries}")
        print(f"  Total Time: {total_time:.2f}ms")
        print(f"  Slow Queries: {total_slow} (>{self.slow_query_threshold}ms)")
        print(f"  Average Time: {total_time/total_queries:.2f}ms")
        
        print(f"\nTop 10 Slowest Queries (by average time):")
        print(f"{'Query Name':<40} {'Count':>8} {'Avg (ms)':>10} {'Max (ms)':>10} {'P95 (ms)':>10}")
        print("-"*80)
        
        for query_name, query_stats in self.get_top_queries(10, 'avg_time'):
            print(f"{query_name:<40} {query_stats['count']:>8} "
                  f"{query_stats['avg_time']:>10.2f} {query_stats['max_time']:>10.2f} "
                  f"{query_stats['p95_time']:>10.2f}")
        
        print("="*80 + "\n")


# Global profiler instance
_global_profiler = QueryProfiler()


def profile_query(query_name: str):
    """
    Convenience decorator using global profiler
    
    Example:
        >>> @profile_query("get_price_history")
        >>> def get_price_history(symbol, days):
        >>>     return fetch_data(symbol, days)
    """
    return _global_profiler.profile(query_name)


def get_profiler() -> QueryProfiler:
    """Get the global profiler instance"""
    return _global_profiler


def print_query_stats():
    """Print statistics from global profiler"""
    _global_profiler.print_summary()


def export_query_stats(filepath: str = "logs/query_profiles.json"):
    """Export statistics from global profiler"""
    _global_profiler.export_to_json(filepath)


if __name__ == "__main__":
    # Example usage
    profiler = QueryProfiler(slow_query_threshold=100.0)
    
    @profiler.profile("example_query")
    def example_query(n):
        time.sleep(n / 1000)  # Simulate query time
        return list(range(n))
    
    # Run some queries
    for i in range(10):
        example_query(50 + i * 10)
    
    # Print summary
    profiler.print_summary()
    
    # Export to JSON
    profiler.export_to_json("example_profiles.json")
