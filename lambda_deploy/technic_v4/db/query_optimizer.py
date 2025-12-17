"""
Query Optimizer
Optimizes database queries for better performance
"""

from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass
import functools
import time
from collections import defaultdict


@dataclass
class QueryOptimization:
    """Record of a query optimization"""
    original_query: str
    optimized_query: str
    optimization_type: str
    estimated_improvement: float
    applied: bool = False


class QueryOptimizer:
    """
    Optimizes database queries for performance
    
    Features:
    - N+1 query detection and fixing
    - Batch operation conversion
    - JOIN optimization
    - Query result caching
    - Query rewriting
    
    Example:
        >>> optimizer = QueryOptimizer()
        >>> 
        >>> # Detect N+1 queries
        >>> optimizer.detect_n_plus_1_queries(query_log)
        >>> 
        >>> # Convert to batch operation
        >>> batch_query = optimizer.convert_to_batch(queries)
        >>> 
        >>> # Cache results
        >>> @optimizer.cache_result(ttl=60)
        >>> def expensive_query():
        >>>     return fetch_data()
    """
    
    def __init__(self):
        """Initialize query optimizer"""
        self.optimizations: List[QueryOptimization] = []
        self.query_cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.batch_size = 100  # Default batch size
    
    def detect_n_plus_1_queries(self, query_log: List[Dict]) -> List[Dict]:
        """
        Detect N+1 query patterns
        
        Args:
            query_log: List of executed queries with metadata
        
        Returns:
            List of detected N+1 patterns
        """
        patterns = []
        
        # Group queries by similarity
        query_groups = defaultdict(list)
        
        for query in query_log:
            # Simple pattern matching (in real implementation, use more sophisticated analysis)
            query_text = query.get('query', '')
            
            # Extract base pattern (remove specific values)
            base_pattern = self._extract_query_pattern(query_text)
            query_groups[base_pattern].append(query)
        
        # Detect N+1: one query followed by N similar queries
        for pattern, queries in query_groups.items():
            if len(queries) > 5:  # Threshold for N+1 detection
                # Check if queries are sequential
                timestamps = [q.get('timestamp', 0) for q in queries]
                if self._are_sequential(timestamps):
                    patterns.append({
                        'pattern': pattern,
                        'count': len(queries),
                        'queries': queries,
                        'recommendation': 'Convert to batch query or use JOIN'
                    })
        
        return patterns
    
    def _extract_query_pattern(self, query: str) -> str:
        """
        Extract base pattern from query by removing specific values
        
        Args:
            query: SQL query string
        
        Returns:
            Base pattern string
        """
        # Simple implementation - replace specific values with placeholders
        import re
        
        # Replace string literals
        pattern = re.sub(r"'[^']*'", "'?'", query)
        
        # Replace numbers
        pattern = re.sub(r'\b\d+\b', '?', pattern)
        
        # Replace IN clauses
        pattern = re.sub(r'IN\s*\([^)]+\)', 'IN (?)', pattern, flags=re.IGNORECASE)
        
        return pattern
    
    def _are_sequential(self, timestamps: List[float], threshold: float = 1.0) -> bool:
        """
        Check if timestamps are sequential (within threshold)
        
        Args:
            timestamps: List of timestamps
            threshold: Maximum time gap in seconds
        
        Returns:
            True if sequential
        """
        if len(timestamps) < 2:
            return False
        
        sorted_times = sorted(timestamps)
        
        for i in range(len(sorted_times) - 1):
            gap = sorted_times[i + 1] - sorted_times[i]
            if gap > threshold:
                return False
        
        return True
    
    def convert_to_batch(self, queries: List[str], batch_size: Optional[int] = None) -> List[str]:
        """
        Convert multiple similar queries to batch operations
        
        Args:
            queries: List of similar queries
            batch_size: Size of each batch (default: self.batch_size)
        
        Returns:
            List of batch queries
        """
        batch_size = batch_size or self.batch_size
        batch_queries = []
        
        # Group queries into batches
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            
            # Extract values from individual queries
            values = [self._extract_query_values(q) for q in batch]
            
            # Create batch query
            if values:
                batch_query = self._create_batch_query(batch[0], values)
                batch_queries.append(batch_query)
        
        return batch_queries
    
    def _extract_query_values(self, query: str) -> List[str]:
        """
        Extract values from a query
        
        Args:
            query: SQL query
        
        Returns:
            List of values
        """
        import re
        
        # Extract values from WHERE clause
        values = re.findall(r"=\s*'([^']+)'", query)
        values.extend(re.findall(r'=\s*(\d+)', query))
        
        return values
    
    def _create_batch_query(self, template_query: str, values_list: List[List[str]]) -> str:
        """
        Create a batch query from template and values
        
        Args:
            template_query: Template query
            values_list: List of value lists
        
        Returns:
            Batch query string
        """
        # Simple implementation - convert to IN clause
        if not values_list:
            return template_query
        
        # Flatten values
        all_values = [v for values in values_list for v in values]
        
        # Replace = with IN
        batch_query = template_query.replace('= ?', f"IN ({', '.join(['?' for _ in all_values])})")
        
        return batch_query
    
    def optimize_joins(self, query: str) -> str:
        """
        Optimize JOIN operations
        
        Args:
            query: SQL query with JOINs
        
        Returns:
            Optimized query
        """
        optimized = query
        
        # Optimization 1: Use INNER JOIN instead of WHERE for joins
        if 'WHERE' in query.upper() and '=' in query:
            # Convert implicit joins to explicit INNER JOINs
            # (Simplified - real implementation would parse SQL properly)
            pass
        
        # Optimization 2: Reorder JOINs (smaller tables first)
        # Would require table statistics
        
        # Optimization 3: Use appropriate JOIN type
        # LEFT JOIN vs INNER JOIN based on requirements
        
        return optimized
    
    def cache_result(self, ttl: int = 60, key_func: Optional[Callable] = None):
        """
        Decorator to cache query results
        
        Args:
            ttl: Time to live in seconds
            key_func: Function to generate cache key from arguments
        
        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                
                # Check cache
                if cache_key in self.query_cache:
                    cached_data, cached_time = self.query_cache[cache_key]
                    
                    # Check if still valid
                    if time.time() - cached_time < ttl:
                        self.cache_hits += 1
                        return cached_data
                
                # Cache miss - execute query
                self.cache_misses += 1
                result = func(*args, **kwargs)
                
                # Store in cache
                self.query_cache[cache_key] = (result, time.time())
                
                return result
            
            return wrapper
        return decorator
    
    def batch_fetch(self, fetch_func: Callable, ids: List[Any], batch_size: Optional[int] = None) -> Dict[Any, Any]:
        """
        Batch fetch data for multiple IDs
        
        Args:
            fetch_func: Function to fetch data (takes list of IDs)
            ids: List of IDs to fetch
            batch_size: Size of each batch
        
        Returns:
            Dictionary mapping ID to result
        """
        batch_size = batch_size or self.batch_size
        results = {}
        
        # Process in batches
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            
            try:
                batch_results = fetch_func(batch_ids)
                
                # Merge results
                if isinstance(batch_results, dict):
                    results.update(batch_results)
                elif isinstance(batch_results, list):
                    # Assume results are in same order as IDs
                    for id_val, result in zip(batch_ids, batch_results):
                        results[id_val] = result
            
            except Exception as e:
                print(f"[OPTIMIZER] Batch fetch error: {e}")
                # Fall back to individual fetches
                for id_val in batch_ids:
                    try:
                        results[id_val] = fetch_func([id_val])
                    except:
                        results[id_val] = None
        
        return results
    
    def optimize_query_string(self, query: str) -> str:
        """
        Optimize a query string
        
        Args:
            query: SQL query
        
        Returns:
            Optimized query
        """
        optimized = query
        
        # Optimization 1: Remove unnecessary DISTINCT
        if 'DISTINCT' in optimized.upper() and 'GROUP BY' in optimized.upper():
            # DISTINCT is redundant with GROUP BY
            optimized = optimized.replace('DISTINCT ', '', 1)
        
        # Optimization 2: Use EXISTS instead of COUNT
        if 'COUNT(*)' in optimized.upper() and '> 0' in optimized:
            # EXISTS is faster than COUNT for existence checks
            optimized = optimized.replace('COUNT(*) > 0', 'EXISTS')
        
        # Optimization 3: Limit result set early
        if 'ORDER BY' in optimized.upper() and 'LIMIT' not in optimized.upper():
            # Add LIMIT if sorting without limit
            # (Would need context to determine appropriate limit)
            pass
        
        return optimized
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.query_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
    def clear_cache(self):
        """Clear the query cache"""
        self.query_cache.clear()
        print(f"[OPTIMIZER] Cleared cache ({self.cache_hits} hits, {self.cache_misses} misses)")
        self.cache_hits = 0
        self.cache_misses = 0
    
    def add_optimization(self, optimization: QueryOptimization):
        """
        Add an optimization record
        
        Args:
            optimization: QueryOptimization instance
        """
        self.optimizations.append(optimization)
    
    def get_optimization_report(self) -> str:
        """
        Generate optimization report
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*80)
        report.append("QUERY OPTIMIZATION REPORT")
        report.append("="*80)
        report.append("")
        
        if not self.optimizations:
            report.append("No optimizations recorded yet.")
            return "\n".join(report)
        
        report.append(f"Total Optimizations: {len(self.optimizations)}")
        report.append(f"Applied: {sum(1 for opt in self.optimizations if opt.applied)}")
        report.append(f"Pending: {sum(1 for opt in self.optimizations if not opt.applied)}")
        report.append("")
        
        # Group by type
        by_type = defaultdict(list)
        for opt in self.optimizations:
            by_type[opt.optimization_type].append(opt)
        
        for opt_type, opts in sorted(by_type.items()):
            report.append(f"Type: {opt_type}")
            report.append("-" * 80)
            
            for opt in opts:
                status = "✓" if opt.applied else "○"
                report.append(f"  {status} Improvement: {opt.estimated_improvement:.1f}x")
                report.append(f"     Original: {opt.original_query[:60]}...")
                report.append(f"     Optimized: {opt.optimized_query[:60]}...")
                report.append("")
        
        # Cache stats
        cache_stats = self.get_cache_stats()
        report.append("Cache Statistics:")
        report.append("-" * 80)
        report.append(f"  Size: {cache_stats['cache_size']} entries")
        report.append(f"  Hits: {cache_stats['cache_hits']}")
        report.append(f"  Misses: {cache_stats['cache_misses']}")
        report.append(f"  Hit Rate: {cache_stats['hit_rate']:.1f}%")
        report.append("")
        
        report.append("="*80)
        return "\n".join(report)


# Global optimizer instance
_global_optimizer = QueryOptimizer()


def get_optimizer() -> QueryOptimizer:
    """Get the global optimizer instance"""
    return _global_optimizer


def cache_query_result(ttl: int = 60):
    """
    Convenience decorator for caching query results
    
    Args:
        ttl: Time to live in seconds
    
    Example:
        >>> @cache_query_result(ttl=300)
        >>> def get_expensive_data(symbol):
        >>>     return fetch_from_api(symbol)
    """
    return _global_optimizer.cache_result(ttl=ttl)


def batch_fetch_data(fetch_func: Callable, ids: List[Any], batch_size: int = 100) -> Dict[Any, Any]:
    """
    Convenience function for batch fetching
    
    Args:
        fetch_func: Function to fetch data
        ids: List of IDs
        batch_size: Batch size
    
    Returns:
        Dictionary of results
    """
    return _global_optimizer.batch_fetch(fetch_func, ids, batch_size)


if __name__ == "__main__":
    # Example usage
    optimizer = QueryOptimizer()
    
    # Example 1: Cache results
    @optimizer.cache_result(ttl=60)
    def expensive_query(symbol):
        print(f"Fetching data for {symbol}...")
        time.sleep(0.1)  # Simulate slow query
        return {"symbol": symbol, "price": 100}
    
    # First call - cache miss
    result1 = expensive_query("AAPL")
    print(f"Result 1: {result1}")
    
    # Second call - cache hit
    result2 = expensive_query("AAPL")
    print(f"Result 2: {result2}")
    
    # Print stats
    print(f"\nCache Stats: {optimizer.get_cache_stats()}")
    
    # Example 2: Batch fetch
    def fetch_symbols(symbol_list):
        return {s: {"price": 100} for s in symbol_list}
    
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    results = optimizer.batch_fetch(fetch_symbols, symbols, batch_size=2)
    print(f"\nBatch Results: {results}")
    
    # Example 3: N+1 detection
    query_log = [
        {"query": "SELECT * FROM users WHERE id = 1", "timestamp": 1.0},
        {"query": "SELECT * FROM posts WHERE user_id = 1", "timestamp": 1.1},
        {"query": "SELECT * FROM posts WHERE user_id = 2", "timestamp": 1.2},
        {"query": "SELECT * FROM posts WHERE user_id = 3", "timestamp": 1.3},
        {"query": "SELECT * FROM posts WHERE user_id = 4", "timestamp": 1.4},
        {"query": "SELECT * FROM posts WHERE user_id = 5", "timestamp": 1.5},
        {"query": "SELECT * FROM posts WHERE user_id = 6", "timestamp": 1.6},
    ]
    
    n_plus_1 = optimizer.detect_n_plus_1_queries(query_log)
    print(f"\nN+1 Patterns Detected: {len(n_plus_1)}")
    for pattern in n_plus_1:
        print(f"  Pattern: {pattern['pattern']}")
        print(f"  Count: {pattern['count']}")
        print(f"  Recommendation: {pattern['recommendation']}")
    
    # Print report
    print(optimizer.get_optimization_report())
