# Phase 3C: Redis Caching - Implementation Started

## Goal
Achieve 2x speedup (90-180 seconds per scan) through aggressive Redis caching of:
- Technical indicators (5 min TTL)
- ML predictions (5 min TTL)  
- Price data (1 hour TTL)
- Scan results (5 min TTL)

## Files Created

### 1. Redis Cache Layer
**File**: `technic_v4/cache/redis_cache.py`
- `RedisCache` class with connection management
- Automatic fallback if Redis unavailable
- Batch get/set operations for efficiency
- Cache decorators for indicators and ML predictions
- Cache warming for top symbols
- Statistics tracking (hit rate, etc.)

**File**: `technic_v4/cache/__init__.py`
- Module exports

### 2. Scanner Integration
**File**: `technic_v4/scanner_core.py` (modified)
- Added Redis import at top
- `REDIS_AVAILABLE` flag set based on connection
- Ready for cache integration in key functions

## Next Steps

### Immediate (to complete Phase 3C):

1. **Integrate cache warming in `run_scan()`**
   - Warm cache for top 500 symbols before scanning
   - Use `redis_cache.warm_cache(symbols, days)`

2. **Cache scan results in `_scan_symbol()`**
   - Check cache before computing
   - Store results after computing
   - Key format: `scan_result:{symbol}:{date}`

3. **Cache indicators in `compute_scores()`**
   - Use `@redis_cache.cache_indicators()` decorator
   - Or manual cache check/set

4. **Cache ML predictions in alpha inference**
   - Modify `alpha_inference.py` to check cache
   - Store predictions with symbol+date key

5. **Add Redis to deployment**
   - Local: Install Redis (`brew install redis` or `apt-get install redis`)
   - Render: Add Redis addon ($10/month)
   - Set `REDIS_URL` environment variable

## Configuration

### Environment Variables
```bash
# Redis connection (optional, defaults to localhost)
REDIS_URL=redis://localhost:6379
# or
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Cache TTLs
- **Indicators**: 300 seconds (5 minutes)
- **ML Predictions**: 300 seconds (5 minutes)
- **Price Data**: 3600 seconds (1 hour)
- **Scan Results**: 300 seconds (5 minutes)

## Expected Performance

### Without Redis (Current - Phase 3B)
- Full scan: 180-360 seconds (3-6 minutes)
- Per-symbol: 0.345s

### With Redis (Phase 3C Target)
- **First scan**: 180-360 seconds (same, cache miss)
- **Subsequent scans**: 90-180 seconds (2x faster, cache hit)
- **Cache hit rate target**: 70-85%

## Benefits

1. **2x Speedup**: Subsequent scans within 5 minutes are 2x faster
2. **Reduced API Load**: Fewer Polygon API calls
3. **Lower Costs**: Less API usage
4. **Better UX**: Faster results for users
5. **Scalability**: Multiple users benefit from shared cache

## Status

✅ **Redis cache layer created**
✅ **Scanner import added**
⏳ **Cache integration pending**
⏳ **Testing pending**
⏳ **Deployment pending**

---

**Next**: Integrate cache checks into `_scan_symbol()` and `run_scan()`
