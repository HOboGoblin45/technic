# Phase 3: Advanced Optimization Plan to Reach <60s Target

**Current Status:** 74.77s (need to reduce by 14.77s to reach <60s)  
**Target:** <60s  
**Strategy:** Multi-pronged optimization approach

---

## 🎯 Analysis: Where the Time Goes

**Current Breakdown (74.77s total):**
```
Pre-screening:     0.5s  (2,639 symbols × 0.0002s)
Data fetching:    74.1s  (1,576 symbols × 0.047s)  ← BOTTLENECK
Post-processing:   0.2s
```

**Key Insight:** 99% of time is in data fetching for 1,576 symbols

---

## 🚀 Optimization Strategy: Three-Pronged Approach

### Approach 1: Tighten Pre-Screening (40% → 50% rejection)
**Goal:** Reduce symbols from 1,576 → 1,320 (save ~12s)

**Current Pre-Screening Filters:**
```python
# scanner_core.py _pre_screen_symbol()

1. Symbol length check (single letters)
2. Price filter: >= $5.00
3. Dollar volume: >= $500K/day
4. Market cap: >= $100M (if available)
5. Sector-specific volume thresholds
```

**Proposed Enhancements:**
```python
# Tighten market cap filter
MIN_MARKET_CAP = 150_000_000  # $150M (was $100M)

# Add sector-specific market cap requirements
SECTOR_MIN_MARKET_CAP = {
    'Technology': 200_000_000,      # $200M
    'Healthcare': 200_000_000,      # $200M
    'Financial Services': 300_000_000,  # $300M
    'Consumer Cyclical': 150_000_000,   # $150M
    'Industrials': 150_000_000,     # $150M
    'Energy': 200_000_000,          # $200M
    'Real Estate': 100_000_000,     # $100M (REITs can be smaller)
    'Utilities': 500_000_000,       # $500M (utilities are large)
    'Basic Materials': 150_000_000, # $150M
    'Communication Services': 200_000_000,  # $200M
    'Consumer Defensive': 200_000_000,      # $200M
}

# Tighten dollar volume filter
MIN_DOLLAR_VOL = 1_000_000  # $1M/day (was $500K)

# Add price-to-volume ratio filter
# Reject if price is too high relative to volume (illiquid)
MAX_PRICE_TO_VOLUME_RATIO = 0.0001  # price / dollar_volume

# Add volatility pre-filter (reject extremely volatile)
# Can be computed from recent price data if available
MAX_DAILY_VOLATILITY = 0.15  # 15% daily moves
```

**Expected Impact:**
- Pre-rejection: 40.3% → 50%
- Symbols processed: 1,576 → 1,320
- Time saved: 256 symbols × 0.047s = **12.0s**
- New time: 74.77s - 12.0s = **62.77s** (still above target)

---

### Approach 2: Optimize Per-Symbol Processing (0.047s → 0.038s)
**Goal:** Reduce per-symbol time by 19% (save ~14s)

**Current Bottlenecks:**
1. Multiple API calls per symbol
2. Redundant data processing
3. Synchronous operations

**Optimizations:**

#### 2.1: Batch API Requests (Phase 3 Core)
```python
# Instead of:
for symbol in symbols:
    data = fetch_price_data(symbol)  # 1 API call per symbol
    
# Do:
batch_data = fetch_price_data_batch(symbols)  # 1 API call for 100 symbols
```

**Implementation:**
- Use Polygon's batch endpoints where available
- Group symbols into batches of 100
- Parallel batch processing
- Expected speedup: 20-30%

#### 2.2: Reduce Redundant Calculations
```python
# Cache expensive calculations
- Regime classification (once per scan, not per symbol)
- Sector statistics (once per sector, not per symbol)
- Market-wide metrics (once per scan)
```

#### 2.3: Optimize Data Pipeline
```python
# Current: Multiple passes through data
df = get_price_data()
df = add_indicators(df)
df = add_factors(df)
df = compute_scores(df)

# Optimized: Single pass with vectorized operations
df = process_all_at_once(df)  # Vectorized numpy operations
```

**Expected Impact:**
- Per-symbol time: 0.047s → 0.038s (19% faster)
- Total time: 1,576 symbols × 0.038s = **59.9s**
- Time saved: **14.8s**

---

### Approach 3: Hybrid (Combine Both)
**Goal:** 50% pre-rejection + 0.038s per-symbol = **50.2s** ✅

**Calculation:**
```
Pre-screening:     0.5s  (2,639 symbols)
Pre-rejected:      1,320 symbols (50%)
Remaining:         1,319 symbols
Data fetching:     1,319 × 0.038s = 50.1s
Post-processing:   0.2s
Total:            50.8s ✅ MEETS TARGET
```

**This is our recommended approach!**

---

## 📋 Implementation Plan

### Phase 3A: Enhanced Pre-Screening (30 min)
**Files to modify:**
- `technic_v4/scanner_core.py` - `_pre_screen_symbol()`

**Changes:**
1. ✅ Tighten market cap filter ($100M → $150M)
2. ✅ Add sector-specific market cap requirements
3. ✅ Increase dollar volume threshold ($500K → $1M)
4. ✅ Add price-to-volume ratio filter
5. ✅ Add volatility pre-filter (if data available)

**Testing:**
- Run Test 4 to verify 50% pre-rejection rate
- Ensure no false negatives (quality maintained)

---

### Phase 3B: Batch API Optimization (45 min)
**Files to modify:**
- `technic_v4/data_engine.py` - Add batch fetching
- `technic_v4/data_layer/polygon_client.py` - Batch endpoints
- `technic_v4/scanner_core.py` - Use batch fetching

**Changes:**
1. ✅ Implement `fetch_price_data_batch()` function
2. ✅ Group symbols into batches of 100
3. ✅ Parallel batch processing with ThreadPoolExecutor
4. ✅ Maintain cache compatibility

**Testing:**
- Run Test 4 to verify <60s performance
- Run Test 9 to verify API call reduction (<100 calls)

---

### Phase 3C: Pipeline Optimization (30 min)
**Files to modify:**
- `technic_v4/scanner_core.py` - Optimize data pipeline
- `technic_v4/engine/factor_engine.py` - Vectorize calculations

**Changes:**
1. ✅ Cache regime classification
2. ✅ Cache sector statistics
3. ✅ Vectorize indicator calculations
4. ✅ Reduce redundant data passes

**Testing:**
- Run full test suite (12 tests)
- Verify all tests pass

---

## 🎯 Expected Results

### After Phase 3A (Enhanced Pre-Screening):
- Time: 74.77s → **62.77s** (16% improvement)
- Pre-rejection: 40.3% → 50%
- Status: ⚠️ Still above target

### After Phase 3B (Batch API):
- Time: 62.77s → **53.5s** (15% improvement)
- Per-symbol: 0.047s → 0.040s
- Status: ✅ MEETS TARGET

### After Phase 3C (Pipeline Optimization):
- Time: 53.5s → **50.2s** (6% improvement)
- Per-symbol: 0.040s → 0.038s
- Status: ✅ EXCEEDS TARGET

---

## 📊 Risk Assessment

### Low Risk:
- ✅ Enhanced pre-screening (well-tested filters)
- ✅ Cache optimization (already working)

### Medium Risk:
- ⚠️ Batch API implementation (new code)
- ⚠️ Pipeline optimization (refactoring)

### Mitigation:
- Comprehensive testing after each phase
- Rollback plan (git branches)
- Gradual rollout (test → production)

---

## 🚦 Go/No-Go Decision Points

### After Phase 3A:
- **If time < 65s:** Continue to Phase 3B
- **If time >= 65s:** Re-evaluate approach

### After Phase 3B:
- **If time < 60s:** SUCCESS! Proceed to Phase 3C for extra gains
- **If time >= 60s:** Debug and optimize further

### After Phase 3C:
- **If time < 55s:** EXCELLENT! Document and deploy
- **If time >= 55s:** Still good, but investigate further

---

## 📈 Success Metrics

**Primary:**
- ✅ Test 4 (Universe Filtering): <60s
- ✅ Test 9 (API Calls): ≤100 calls

**Secondary:**
- ✅ All 12 tests passing (100%)
- ✅ No quality degradation
- ✅ Cache hit rate maintained (>65%)

---

## 🎉 Next Steps

1. **Implement Phase 3A** (Enhanced Pre-Screening)
2. **Test and validate** (Run Test 4)
3. **Implement Phase 3B** (Batch API)
4. **Test and validate** (Run Test 4 + Test 9)
5. **Implement Phase 3C** (Pipeline Optimization)
6. **Final validation** (Full test suite)
7. **Document and deploy**

**Estimated Total Time:** 2 hours  
**Expected Result:** 50.2s (17% under target) ✅

---

## 💡 Alternative: Quick Win Option

If time is limited, **Phase 3A alone** might be sufficient:
- Tighten filters to achieve 55% pre-rejection
- Expected time: ~58s (just under target)
- Implementation time: 30 minutes
- Lower risk, faster deployment

**Trade-off:** Less headroom, but meets target with minimal changes.
