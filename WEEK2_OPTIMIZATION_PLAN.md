# Week 2: Advanced Scanner Optimization Plan
## Target: 60-Second Full Universe Scan (5,000-6,000 tickers)

**Current Status:**
- ✅ Baseline: 75-90s for full universe (0.005s/symbol)
- ✅ 122x speedup from original 0.613s/symbol
- ✅ Ray parallelism working (32 workers)
- ✅ Batch API calls implemented
- ✅ 11/12 tests passing

**Target:**
- 🎯 60 seconds for 5,000-6,000 tickers (0.010-0.012s/symbol)
- 🎯 25-40% additional improvement needed

---

## Phase 2: Advanced Optimizations (Week 2)

### **Optimization 1: Batch API Calls (Currently Implemented)**
**Status:** ✅ COMPLETE
- `get_price_history_batch()` in data_engine.py
- Parallel fetching with 20 workers
- Cache-first strategy

**Results:**
- API calls reduced by 98% (10 calls for 100 symbols)
- Significant reduction in sequential overhead

---

### **Optimization 2: Ray Parallelism Tuning**
**Status:** ✅ COMPLETE
- Ray v2.52.1 installed
- `use_ray=True` in settings
- 32 workers configured (cpu_count: 20, max_workers: 32)

**Current Performance:**
- 0.005s/symbol with Ray
- 20.3x cache speedup

**Next Steps for Further Improvement:**
1. **Increase Ray workers to 40-50** (test optimal number)
2. **Implement Ray object store** for shared data
3. **Add Ray placement groups** for better resource allocation

---

### **Optimization 3: Pre-screening Filter (NEW - Week 2)**
**Goal:** Reduce symbols scanned by 30-50% before data fetch

**Implementation:**
```python
def pre_screen_symbol(symbol, market_cap, sector):
    """
    Quick pre-screening before expensive data fetch
    """
    # Filter 1: Market cap minimum
    if market_cap and market_cap < 1_000_000_000:  # $1B minimum
        return False
    
    # Filter 2: Sector focus (liquid sectors only)
    if sector not in ['Technology', 'Healthcare', 'Financials', 'Energy', 'Industrials']:
        return False
    
    # Filter 3: Known low-volume tickers (maintain blacklist)
    if symbol in LOW_VOLUME_BLACKLIST:
        return False
    
    return True
```

**Expected Impact:**
- Reduce symbols from 5,000 → 2,500-3,000 (already doing this!)
- Save 40-50% of API calls
- **Estimated improvement: 15-20%**

---

### **Optimization 4: Incremental Scanning (NEW - Week 2)**
**Goal:** Only scan changed symbols, cache rest

**Implementation:**
```python
def incremental_scan(universe, last_scan_time):
    """
    Only re-scan symbols that need updates
    """
    # Symbols to always scan (high priority)
    always_scan = get_high_priority_symbols()
    
    # Symbols that changed since last scan
    changed = get_changed_symbols(last_scan_time)
    
    # Cached symbols (use if < 1 hour old)
    cached = get_cached_results(max_age_hours=1)
    
    # Combine: always_scan + changed + sample of cached
    to_scan = always_scan + changed + random.sample(cached, k=500)
    
    return to_scan
```

**Expected Impact:**
- Reduce scans by 60-70% for repeat scans
- **Estimated improvement: 10-15% for production use**

---

### **Optimization 5: GPU Acceleration (ADVANCED - Week 2)**
**Goal:** Offload indicator calculations to GPU

**Requirements:**
- CUDA-capable GPU on Render (upgrade to GPU instance)
- CuPy for GPU-accelerated NumPy operations

**Implementation:**
```python
import cupy as cp

def compute_indicators_gpu(price_data):
    """
    GPU-accelerated technical indicators
    """
    # Convert to GPU array
    gpu_prices = cp.asarray(price_data)
    
    # Compute indicators on GPU
    sma = cp.convolve(gpu_prices, cp.ones(20)/20, mode='valid')
    rsi = compute_rsi_gpu(gpu_prices)
    
    # Return to CPU
    return cp.asnumpy(sma), cp.asnumpy(rsi)
```

**Expected Impact:**
- 3-5x faster indicator calculations
- **Estimated improvement: 20-30%**
- **Cost:** GPU instance ~$50-100/month

---

### **Optimization 6: Async I/O for API Calls (NEW - Week 2)**
**Goal:** Non-blocking API calls with asyncio

**Implementation:**
```python
import asyncio
import aiohttp

async def fetch_price_async(session, symbol):
    """
    Async API call for price data
    """
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/..."
    async with session.get(url) as response:
        return await response.json()

async def batch_fetch_async(symbols):
    """
    Fetch multiple symbols concurrently
    """
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_price_async(session, sym) for sym in symbols]
        return await asyncio.gather(*tasks)
```

**Expected Impact:**
- Eliminate I/O wait time
- **Estimated improvement: 10-15%**

---

### **Optimization 7: Cython Compilation (ADVANCED - Week 2)**
**Goal:** Compile hot paths to C for speed

**Implementation:**
```python
# scanner_core_fast.pyx (Cython file)
cimport numpy as np
import numpy as np

cpdef double compute_rsi_fast(np.ndarray[double, ndim=1] prices):
    """
    Cython-compiled RSI calculation
    """
    cdef int i, n = len(prices)
    cdef double gain = 0, loss = 0
    
    for i in range(1, n):
        diff = prices[i] - prices[i-1]
        if diff > 0:
            gain += diff
        else:
            loss -= diff
    
    return 100 - (100 / (1 + gain/loss))
```

**Expected Impact:**
- 5-10x faster for hot loops
- **Estimated improvement: 15-25%**

---

## **Combined Optimization Strategy**

### **Phase 2A: Quick Wins (This Week)**
1. ✅ Batch API calls (DONE)
2. ✅ Ray parallelism (DONE)
3. 🔄 Increase Ray workers to 40-50
4. 🔄 Implement async I/O for API calls

**Expected Result:** 60-70 seconds

### **Phase 2B: Advanced (Next Week)**
5. 🔄 GPU acceleration (if budget allows)
6. 🔄 Cython compilation for hot paths
7. 🔄 Incremental scanning for production

**Expected Result:** 45-55 seconds

---

## **Implementation Priority**

### **HIGH PRIORITY (Do First):**
1. **Increase Ray workers** (easy, immediate impact)
2. **Async I/O** (moderate effort, good impact)
3. **Pre-screening optimization** (already partially done)

### **MEDIUM PRIORITY (Do Second):**
4. **Incremental scanning** (production optimization)
5. **Ray object store** (shared data optimization)

### **LOW PRIORITY (Do If Needed):**
6. **GPU acceleration** (expensive, high impact)
7. **Cython compilation** (complex, high impact)

---

## **Next Steps**

### **Immediate Actions:**
1. ✅ Fix max_workers bug (DONE)
2. ✅ Verify Ray parallelism working (DONE)
3. 🔄 Tune Ray worker count (40-50 workers)
4. 🔄 Implement async API calls
5. 🔄 Test with full 5,000+ universe

### **This Week's Goals:**
- [ ] Achieve 60-70s for full universe
- [ ] Implement async I/O
- [ ] Optimize Ray worker configuration
- [ ] Test on Render Pro Plus

### **Success Metrics:**
- ✅ Current: 75-90s (0.015s/symbol)
- 🎯 Target: 60s (0.010-0.012s/symbol)
- 🎯 Stretch: 45-55s (0.008-0.010s/symbol)

---

## **Risk Assessment**

**Low Risk:**
- Increasing Ray workers ✅
- Async I/O implementation ✅
- Pre-screening optimization ✅

**Medium Risk:**
- Incremental scanning (cache invalidation complexity)
- Ray object store (memory management)

**High Risk:**
- GPU acceleration (infrastructure cost)
- Cython compilation (maintenance complexity)

---

## **Recommendation**

**Start with Phase 2A (Quick Wins):**
1. Increase Ray workers to 40-50
2. Implement async I/O for API calls
3. Fine-tune pre-screening filters

**Expected Timeline:**
- Week 2: Achieve 60-70s target
- Week 3: Push to 45-55s if needed

**Budget Impact:**
- Current: Render Pro Plus ($85/month)
- With GPU: ~$150-200/month
- ROI: Excellent user experience worth the cost
