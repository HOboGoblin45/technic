# Task 7: Query Optimization - Implementation Plan

## Overview

Optimize database queries and data access patterns to achieve 10x speedup through indexing, query optimization, and efficient data structures.

**Time Estimate:** 8 hours  
**Target:** 10x query speedup  
**Current:** Baseline query performance

---

## Objectives

1. **Identify Slow Queries** - Profile and benchmark current performance
2. **Add Database Indexes** - Strategic indexing for common queries
3. **Optimize Query Patterns** - Reduce N+1 queries, use batch operations
4. **Implement Query Caching** - Cache expensive query results
5. **Add Connection Pooling** - Reuse database connections efficiently

---

## Current Performance Issues

### Identified Bottlenecks

1. **Symbol Lookups** - Linear scans without indexes
2. **Historical Data Queries** - Full table scans
3. **Aggregation Queries** - Inefficient grouping
4. **N+1 Query Problem** - Multiple queries in loops
5. **No Query Result Caching** - Repeated identical queries

### Performance Baseline

```
Symbol lookup: ~500ms (no index)
Historical data: ~2000ms (full scan)
Aggregations: ~1500ms (no optimization)
Batch operations: ~5000ms (N+1 queries)
```

**Target:** All queries <100ms (10x improvement)

---

## Implementation Strategy

### Phase 1: Profiling & Analysis (2 hours)

**1.1 Query Profiling**
- Instrument all database queries
- Measure execution times
- Identify top 10 slowest queries
- Analyze query patterns

**1.2 Bottleneck Analysis**
- Find N+1 query patterns
- Identify missing indexes
- Detect inefficient joins
- Review data access patterns

### Phase 2: Index Optimization (2 hours)

**2.1 Strategic Indexing**
- Add indexes on frequently queried columns
- Composite indexes for multi-column queries
- Covering indexes for common SELECT patterns
- Partial indexes for filtered queries

**2.2 Index Types**
- B-tree indexes (default, most queries)
- Hash indexes (equality comparisons)
- GiST indexes (geometric/full-text)
- Partial indexes (WHERE clause optimization)

### Phase 3: Query Optimization (3 hours)

**3.1 Batch Operations**
- Replace N+1 queries with batch fetches
- Use IN clauses for multiple IDs
- Implement bulk inserts/updates
- Optimize JOIN operations

**3.2 Query Rewriting**
- Simplify complex queries
- Use CTEs for readability
- Optimize subqueries
- Add query hints where needed

**3.3 Data Structure Optimization**
- Use appropriate data types
- Normalize/denormalize strategically
- Add computed columns
- Implement materialized views

### Phase 4: Testing & Validation (1 hour)

**4.1 Performance Testing**
- Benchmark before/after
- Measure query execution times
- Test with realistic data volumes
- Validate correctness

**4.2 Load Testing**
- Test concurrent queries
- Measure throughput
- Check resource usage
- Identify remaining bottlenecks

---

## Technical Implementation

### 1. Query Profiler

```python
class QueryProfiler:
    """Profile database query performance"""
    
    def profile_query(self, query: str, params: dict) -> QueryStats:
        """Profile a single query execution"""
        
    def get_slow_queries(self, threshold_ms: int = 100) -> List[QueryStats]:
        """Get queries slower than threshold"""
        
    def generate_report(self) -> str:
        """Generate performance report"""
```

### 2. Index Manager

```python
class IndexManager:
    """Manage database indexes"""
    
    def create_index(self, table: str, columns: List[str], index_type: str = "btree"):
        """Create database index"""
        
    def analyze_missing_indexes(self) -> List[IndexRecommendation]:
        """Recommend missing indexes"""
        
    def optimize_existing_indexes(self):
        """Optimize existing indexes"""
```

### 3. Query Optimizer

```python
class QueryOptimizer:
    """Optimize query patterns"""
    
    def batch_fetch(self, ids: List[int], table: str) -> List[dict]:
        """Batch fetch multiple records"""
        
    def optimize_joins(self, query: str) -> str:
        """Optimize JOIN operations"""
        
    def add_query_hints(self, query: str) -> str:
        """Add performance hints"""
```

---

## Optimization Strategies

### Strategy 1: Add Indexes

**Symbol Lookups:**
```sql
-- Before: Full table scan
SELECT * FROM symbols WHERE symbol = 'AAPL';  -- 500ms

-- After: Index lookup
CREATE INDEX idx_symbols_symbol ON symbols(symbol);
SELECT * FROM symbols WHERE symbol = 'AAPL';  -- 5ms (100x faster)
```

### Strategy 2: Batch Operations

**N+1 Query Problem:**
```python
# Before: N+1 queries
for symbol in symbols:  # 100 symbols
    data = db.query(f"SELECT * FROM prices WHERE symbol = '{symbol}'")
# Total: 100 queries × 50ms = 5000ms

# After: Single batch query
symbols_str = ','.join(f"'{s}'" for s in symbols)
data = db.query(f"SELECT * FROM prices WHERE symbol IN ({symbols_str})")
# Total: 1 query × 100ms = 100ms (50x faster)
```

### Strategy 3: Query Result Caching

**Repeated Queries:**
```python
# Before: Query every time
def get_symbol_data(symbol):
    return db.query(f"SELECT * FROM symbols WHERE symbol = '{symbol}'")

# After: Cache results
@lru_cache(maxsize=1000)
def get_symbol_data(symbol):
    return db.query(f"SELECT * FROM symbols WHERE symbol = '{symbol}'")
```

### Strategy 4: Optimize Aggregations

**Slow Aggregations:**
```sql
-- Before: Full table scan + grouping
SELECT sector, COUNT(*) FROM symbols GROUP BY sector;  -- 1500ms

-- After: Indexed grouping
CREATE INDEX idx_symbols_sector ON symbols(sector);
SELECT sector, COUNT(*) FROM symbols GROUP BY sector;  -- 50ms (30x faster)
```

### Strategy 5: Connection Pooling

**Connection Overhead:**
```python
# Before: New connection each query
for query in queries:
    conn = create_connection()
    result = conn.execute(query)
    conn.close()
# Overhead: 100ms per connection

# After: Reuse connections
pool = ConnectionPool(max_connections=20)
for query in queries:
    with pool.get_connection() as conn:
        result = conn.execute(query)
# Overhead: 1ms per query (100x faster)
```

---

## Expected Improvements

### Query Performance

| Query Type | Before | After | Improvement |
|------------|--------|-------|-------------|
| Symbol Lookup | 500ms | 5ms | 100x |
| Historical Data | 2000ms | 100ms | 20x |
| Aggregations | 1500ms | 50ms | 30x |
| Batch Operations | 5000ms | 100ms | 50x |
| **Average** | **2250ms** | **64ms** | **35x** |

### System Impact

- **API Response Time:** 2100ms → 200ms (10x faster)
- **Throughput:** 10 req/s → 100 req/s (10x increase)
- **Database Load:** -80% reduction
- **Memory Usage:** +10% (caching overhead)

---

## Implementation Checklist

### Phase 1: Profiling (2h)
- [ ] Create query profiler
- [ ] Instrument all queries
- [ ] Collect baseline metrics
- [ ] Identify top 10 slow queries
- [ ] Analyze query patterns

### Phase 2: Indexing (2h)
- [ ] Create index manager
- [ ] Add symbol lookup indexes
- [ ] Add timestamp indexes
- [ ] Add composite indexes
- [ ] Verify index usage

### Phase 3: Optimization (3h)
- [ ] Implement batch operations
- [ ] Fix N+1 query patterns
- [ ] Add query result caching
- [ ] Optimize JOIN operations
- [ ] Add connection pooling

### Phase 4: Testing (1h)
- [ ] Benchmark all queries
- [ ] Validate correctness
- [ ] Load testing
- [ ] Document improvements

---

## Risk Mitigation

### Risk 1: Index Overhead
- **Impact:** Slower writes, more storage
- **Mitigation:** Strategic indexing, monitor write performance
- **Fallback:** Remove unused indexes

### Risk 2: Cache Invalidation
- **Impact:** Stale data
- **Mitigation:** TTL-based expiration, manual invalidation
- **Fallback:** Shorter TTL, disable caching

### Risk 3: Connection Pool Exhaustion
- **Impact:** Blocked queries
- **Mitigation:** Monitor pool usage, adjust size
- **Fallback:** Increase pool size, add queuing

---

## Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Average Query Time | <100ms | Query profiler |
| P95 Query Time | <200ms | Query profiler |
| Throughput | 100 req/s | Load testing |
| Database CPU | <50% | System monitoring |
| Cache Hit Rate | >80% | Cache stats |

---

## Timeline

### Day 1 (4 hours)
- Hour 1-2: Profiling & analysis
- Hour 3-4: Index optimization

### Day 2 (4 hours)
- Hour 1-3: Query optimization
- Hour 4: Testing & validation

---

## Files to Create

1. `technic_v4/db/query_profiler.py` - Query profiling
2. `technic_v4/db/index_manager.py` - Index management
3. `technic_v4/db/query_optimizer.py` - Query optimization
4. `technic_v4/db/connection_pool.py` - Connection pooling
5. `scripts/optimize_queries.py` - Optimization script
6. `test_query_optimization.py` - Tests
7. `PATH1_TASK7_COMPLETE.md` - Documentation

---

**Ready to begin implementation!** 🚀

**Target:** 10x query speedup  
**Expected:** 35x average improvement  
**Timeline:** 8 hours
