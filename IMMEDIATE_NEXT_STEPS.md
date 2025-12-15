# Immediate Next Steps - Scanner Optimization to 60s

## Current Status
- ✅ Backend: 95% complete, 75-90s scan time
- ✅ Critical bug fixed (max_workers)
- ✅ 11/12 tests passing
- ✅ All dependencies in requirements.txt (including scipy)
- ✅ Git LFS disabled
- ✅ Code pushed to main

## Priority: Backend Performance (This Week)

### Goal: 75-90s → 60s (then → 45-55s)

---

## PHASE 1: Immediate Optimizations (Today/Tomorrow)

### 1. Increase Ray Workers (EASY - 30 min)
**Current:** 32 workers  
**Target:** 50 workers

**File:** `technic_v4/config/settings.py`
```python
max_workers: int = 50  # Increased from 32
```

**Expected Impact:** 15-20% improvement → **60-70s**

### 2. Test Current Performance on Render (1 hour)
- Wait for Render deployment to complete
- SSH or use Render shell to run:
  ```bash
  python test_scanner_optimization_thorough.py
  ```
- Verify actual production performance
- Document baseline on Render Pro Plus

### 3. Implement Async API Calls (2-3 hours)
**File:** `technic_v4/data_layer/polygon_client.py`

Add async batch fetching:
```python
import asyncio
import aiohttp

async def fetch_prices_async(symbols: list, days: int):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_single_async(session, sym, days) for sym in symbols]
        return await asyncio.gather(*tasks)
```

**Expected Impact:** 10-15% improvement → **50-60s**

---

## PHASE 2: Advanced Optimizations (Next 2-3 Days)

### 4. Redis Integration (Already Implemented!)
- Redis cache code exists in `technic_v4/cache/redis_cache.py`
- Just needs Render Redis add-on configured
- Follow `RENDER_REDIS_SETUP.md`

**Expected Impact:** 30-40% on repeat scans → **20-30s**

### 5. Pre-screening Optimization (1-2 hours)
**Already partially done** - smart filtering reduces 5,277 → 2,639 symbols

Enhance further:
```python
def enhanced_pre_screen(symbol, market_cap, sector, avg_volume):
    # Market cap filter
    if market_cap < 1_000_000_000:  # $1B minimum
        return False
    
    # Volume filter (already doing this)
    if avg_volume < 1_000_000:  # 1M shares/day minimum
        return False
    
    # Sector focus (already doing this)
    if sector not in LIQUID_SECTORS:
        return False
    
    return True
```

**Expected Impact:** 5-10% improvement

---

## PHASE 3: Production Validation (End of Week)

### 6. Full Universe Test on Render
- Run scan with max_symbols=6000
- Measure actual time
- Verify cache hit rates
- Check memory usage

### 7. Load Testing
- Simulate 5-10 concurrent scans
- Verify no degradation
- Check API rate limits

---

## TIMELINE

**Day 1 (Today):**
- ✅ Fix Git LFS (DONE)
- ✅ Add scipy to requirements (DONE)
- ✅ Push to main (DONE)
- ⏳ Wait for Render deployment
- 🔄 Verify deployment successful

**Day 2 (Tomorrow):**
- Increase Ray workers to 50
- Test on Render
- Implement async API calls
- **Target: 60s achieved**

**Day 3-4:**
- Configure Redis on Render
- Test Redis caching
- **Target: 20-30s on repeat scans**

**Day 5:**
- Full universe validation
- Load testing
- Document final performance

---

## SUCCESS METRICS

**Minimum (Beta Ready):**
- First scan: <90s ✅ (already achieved)
- Repeat scans: <60s 🎯 (with Redis)

**Stretch Goal:**
- First scan: <60s
- Repeat scans: <30s

**Ideal:**
- First scan: <45s
- Repeat scans: <20s

---

## AFTER BACKEND OPTIMIZATION

Once scanner is optimized (60s target met), focus shifts to:

1. **UI/UX Overhaul** (Critical Path for Beta)
   - Refactor Flutter monolith
   - Implement modern Material 3 design
   - Expose MERIT scores in UI
   - Add onboarding flow

2. **Feature Integration**
   - MERIT score display
   - AI Copilot integration
   - Options strategy UI

3. **Beta Preparation**
   - TestFlight setup
   - Privacy policy
   - Beta tester onboarding

---

## CURRENT FOCUS

**Right now:** Wait for Render deployment to complete, then verify it's working.

**Next:** Implement Day 2 optimizations (Ray workers + async I/O).

**Goal:** Hit 60s target by end of week, then shift to UI work.
