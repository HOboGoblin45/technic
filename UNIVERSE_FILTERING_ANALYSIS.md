# Universe Filtering Optimization Analysis

## Current Performance
- **Test 4 Result**: 152.15s for 2,648 symbols
- **Target**: <60s (60% improvement needed)
- **Per-symbol time**: 0.057s/symbol (152.15s / 2,648)
- **Target per-symbol**: 0.023s/symbol (60s / 2,648)

## Bottleneck Analysis

### 1. **_smart_filter_universe() - Already Implemented** ✅
**Location**: Lines 172-227
**Current Impact**: Reduces universe by 49.8% (5,277 → 2,648)
**Status**: Working well, but can be enhanced

### 2. **_passes_basic_filters() - MAJOR BOTTLENECK** ⚠️
**Location**: Lines 1046-1084
**Issues**:
- Called for EVERY symbol (2,648 times)
- Computes dollar volume with `.tail(40).mean()` - expensive
- Calculates price volatility with `.std()` - expensive
- Runs AFTER data is fetched (wasteful)

**Current Logic**:
```python
def _passes_basic_filters(df: pd.DataFrame) -> bool:
    # ... checks after fetching full history ...
    avg_dollar_vol = float((df["Close"] * df["Volume"]).tail(40).mean())
    if avg_dollar_vol < MIN_DOLLAR_VOL:  # $500K
        return False
    # ... more expensive checks ...
```

### 3. **_scan_symbol() - Sequential Processing** ⚠️
**Location**: Lines 1087-1186
**Issues**:
- Fetches full history for every symbol
- Computes indicators even if symbol will be rejected
- No early rejection before expensive operations

### 4. **_process_symbol() - Wrapper Overhead** ⚠️
**Location**: Lines 1189-1220
**Issues**:
- Adds extra function call overhead
- Duplicates some logic from _scan_symbol()

### 5. **Smart Filter Can Be More Aggressive** 💡
**Location**: Lines 172-227
**Current Filters**:
- Invalid tickers (1-5 chars, alpha only)
- Liquid sectors (8 sectors kept)
- Leveraged ETFs (6 patterns excluded)

**Missing Filters** (can add):
- Market cap pre-filter (from universe data)
- Known penny stocks
- Known illiquid symbols
- Sector-specific volume thresholds

## Optimization Strategy

### Phase 1: Enhanced Pre-filtering (Fastest Wins)
**Goal**: Reduce universe from 2,648 → 1,500 symbols (43% reduction)
**Expected Impact**: 152s → 86s (43% faster)

1. **Add market cap filter to smart_filter**
   - Filter out micro-caps (<$50M) BEFORE scanning
   - Use universe data if available
   
2. **Add volume-based pre-filter**
   - Estimate daily volume from universe metadata
   - Remove symbols likely to fail liquidity check
   
3. **Sector-specific thresholds**
   - Different volume requirements per sector
   - Tech/Healthcare: higher volume needed
   - Utilities/REITs: lower volume acceptable

### Phase 2: Early Rejection in _scan_symbol (Medium Impact)
**Goal**: Reject symbols faster, before expensive operations
**Expected Impact**: 86s → 60s (30% faster)

1. **Move basic filters BEFORE indicator computation**
   ```python
   def _scan_symbol(...):
       df = get_price_history(...)
       
       # EARLY REJECTION (before compute_scores)
       if not _quick_filters(df):  # NEW: fast checks only
           return None
       
       # Only compute if passed quick filters
       scored = compute_scores(df, ...)
   ```

2. **Create _quick_filters() function**
   - Price check (last close)
   - Bar count check
   - Simple volume check (last 5 days avg)
   - Skip expensive .tail(40).mean()

### Phase 3: Parallel Optimization (If Needed)
**Goal**: Better utilize 32 workers
**Expected Impact**: Additional 10-20% if needed

1. **Batch processing**
   - Group symbols by sector
   - Process sectors in parallel
   
2. **Ray optimization**
   - Enable Ray for true parallelism
   - Better than threadpool for CPU-bound work

## Implementation Priority

### HIGH PRIORITY (Implement First)
1. ✅ Enhanced smart_filter with market cap
2. ✅ Volume-based pre-filter
3. ✅ Early rejection in _scan_symbol
4. ✅ Quick filters function

### MEDIUM PRIORITY (If Still Needed)
5. Sector-specific thresholds
6. Batch processing optimization

### LOW PRIORITY (Only If Desperate)
7. Ray parallelism (already have threadpool)
8. Caching universe metadata

## Expected Results

**Conservative Estimate**:
- Phase 1: 152s → 90s (40% improvement)
- Phase 2: 90s → 55s (39% improvement)
- **Total**: 152s → 55s (64% improvement) ✓ MEETS TARGET

**Optimistic Estimate**:
- Phase 1: 152s → 80s (47% improvement)
- Phase 2: 80s → 45s (44% improvement)
- **Total**: 152s → 45s (70% improvement) ✓ EXCEEDS TARGET

## Next Steps

1. Implement Phase 1 optimizations
2. Test with full universe (2,648 symbols)
3. Measure improvement
4. Implement Phase 2 if needed
5. Re-run Test 4 to validate <60s target

---
**Status**: Analysis Complete - Ready for Implementation
**Estimated Time**: 30-45 minutes for Phase 1+2
**Risk**: Low (all changes are additive/defensive)
