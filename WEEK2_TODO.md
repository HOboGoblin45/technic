# Week 2 Scanner Optimization TODO
## Target: 60-Second Full Universe Scan

**Current Performance:** 75-90s for 5,000-6,000 tickers (0.015s/symbol)  
**Week 2 Target:** 60s for 5,000-6,000 tickers (0.010-0.012s/symbol)  
**Improvement Needed:** 25-40% faster

---

## ✅ WEEK 1 COMPLETE (Baseline Established)

### Achievements:
- [x] Fixed critical max_workers bug
- [x] Implemented batch API calls (98% reduction)
- [x] Enabled Ray parallelism (32 workers)
- [x] Achieved 0.005s/symbol performance
- [x] 11/12 tests passing (91.7%)
- [x] 122x speedup from baseline
- [x] **MET 90-SECOND GOAL!**

---

## 🚀 WEEK 2 TASKS

### Phase 2A: Quick Wins (Priority 1)

#### 1. Ray Worker Optimization
- [x] Increase max_workers from 32 to 50 in settings.py
- [ ] Test with different worker counts:
  - [ ] Test with 40 workers
  - [ ] Test with 50 workers  
  - [ ] Test with 60 workers
- [ ] Find optimal configuration for Render Pro Plus
- [ ] Document performance at each level

**Expected Impact:** 10-15% improvement  
**Estimated Time:** 2-3 hours

#### 2. Async I/O Implementation
- [ ] Add aiohttp to requirements.txt
- [ ] Create async API wrapper in data_engine.py
- [ ] Implement `fetch_price_async()` function
- [ ] Implement `batch_fetch_async()` with batching
- [ ] Add sync wrapper for compatibility
- [ ] Test async vs sync performance
- [ ] Update batch API to use async

**Expected Impact:** 15-20% improvement  
**Estimated Time:** 4-6 hours

#### 3. Full Universe Testing
- [ ] Run scan with 2,500 symbols
- [ ] Run scan with 5,000 symbols
- [ ] Run scan with 6,000 symbols
- [ ] Measure actual performance
- [ ] Compare to projections
- [ ] Document results

**Expected Impact:** Validation of improvements  
**Estimated Time:** 2-3 hours

---

### Phase 2B: Advanced Optimizations (Priority 2)

#### 4. Ray Object Store
- [ ] Implement SharedDataStore actor
- [ ] Store market regime data in object store
- [ ] Store macro indicators in object store
- [ ] Update scanner to use shared data
- [ ] Test memory usage
- [ ] Benchmark performance improvement

**Expected Impact:** 5-10% improvement  
**Estimated Time:** 3-4 hours

#### 5. Enhanced Pre-screening
- [ ] Review current pre-screening logic
- [ ] Add market cap filter ($500M minimum)
- [ ] Add volume filter (1M shares/day)
- [ ] Maintain low-volume blacklist
- [ ] Test filtering effectiveness
- [ ] Measure symbols filtered

**Expected Impact:** 5-10% improvement  
**Estimated Time:** 2-3 hours

#### 6. Incremental Scanning
- [ ] Implement cache age tracking
- [ ] Create high-priority symbol list
- [ ] Add changed symbol detection
- [ ] Implement smart sampling logic
- [ ] Test incremental vs full scan
- [ ] Document cache strategy

**Expected Impact:** 10-15% for repeat scans  
**Estimated Time:** 4-5 hours

---

## 📊 TESTING PLAN

### Test 1: Ray Worker Scaling
```bash
# Test with 40 workers
export TECHNIC_MAX_WORKERS=40
python test_scanner_optimization_thorough.py

# Test with 50 workers
export TECHNIC_MAX_WORKERS=50
python test_scanner_optimization_thorough.py

# Test with 60 workers
export TECHNIC_MAX_WORKERS=60
python test_scanner_optimization_thorough.py
```

**Success Criteria:**
- All tests pass
- Performance improves with more workers
- Memory usage stays under 6GB
- No crashes or errors

### Test 2: Async I/O Performance
```python
# Compare sync vs async
import time
from implement_week2_optimizations import get_prices_async_wrapper

symbols = ['AAPL', 'MSFT', 'GOOGL', ...] * 10  # 100 symbols

# Sync baseline
start = time.time()
sync_results = batch_fetch_sync(symbols)
sync_time = time.time() - start

# Async test
start = time.time()
async_results = get_prices_async_wrapper(symbols, api_key)
async_time = time.time() - start

print(f"Sync: {sync_time:.2f}s")
print(f"Async: {async_time:.2f}s")
print(f"Speedup: {sync_time/async_time:.2f}x")
```

**Success Criteria:**
- Async is 1.5-2x faster than sync
- All data fetched correctly
- No timeout errors
- API rate limits respected

### Test 3: Full Universe Scan
```python
from technic_v4.scanner_core import run_scan, ScanConfig
import time

# Test with full universe
config = ScanConfig(max_symbols=6000)
start = time.time()
df, msg = run_scan(config)
elapsed = time.time() - start

print(f"Scanned {len(df)} symbols in {elapsed:.2f}s")
print(f"Performance: {elapsed/len(df):.4f}s/symbol")
print(f"Projected 5K: {(elapsed/len(df))*5000:.2f}s")
```

**Success Criteria:**
- Completes without errors
- Time < 70 seconds for 6,000 symbols
- Results quality maintained
- Memory usage < 7GB

---

## 🎯 SUCCESS METRICS

### Week 2 Targets:
- [ ] 5,000 symbols: <60 seconds ✅
- [ ] 6,000 symbols: <70 seconds ✅
- [ ] API calls: <100 ✅
- [ ] Cache hit rate: >60% ✅
- [ ] Memory usage: <7GB ✅
- [ ] All tests passing ✅

### Performance Breakdown:
- **Current:** 0.015s/symbol (75-90s for 5-6K)
- **Week 2 Target:** 0.010-0.012s/symbol (60-70s for 5-6K)
- **Improvement:** 25-40% faster

---

## 📝 IMPLEMENTATION CHECKLIST

### Step 1: Increase Ray Workers ✅
- [x] Update settings.py (max_workers: 50)
- [ ] Test performance with 50 workers
- [ ] Verify no memory issues
- [ ] Document optimal worker count

### Step 2: Async I/O
- [ ] Add aiohttp to requirements.txt
- [ ] Implement async functions in implement_week2_optimizations.py
- [ ] Integrate with data_engine.py
- [ ] Test async performance
- [ ] Rollback if issues

### Step 3: Testing & Validation
- [ ] Run test suite with new optimizations
- [ ] Full universe scan test
- [ ] Performance benchmarking
- [ ] Quality validation
- [ ] Memory profiling

### Step 4: Documentation
- [ ] Update OPTIMIZATION_SUMMARY.md
- [ ] Create WEEK2_RESULTS.md
- [ ] Update performance metrics
- [ ] Document lessons learned

---

## 🔧 ROLLBACK PLAN

If optimizations cause issues:

1. **Revert Ray workers:**
   ```python
   # In settings.py
   max_workers: int = field(default=32)  # Back to Week 1 value
   ```

2. **Disable async I/O:**
   - Comment out async code
   - Use original sync batch API

3. **Re-run tests:**
   ```bash
   python test_scanner_optimization_thorough.py
   ```

---

## 📈 EXPECTED TIMELINE

- **Day 1:** Increase Ray workers, test (2-3 hours)
- **Day 2:** Implement async I/O (4-6 hours)
- **Day 3:** Full testing & validation (3-4 hours)
- **Day 4:** Documentation & deployment (2-3 hours)

**Total:** 11-16 hours over 4 days

---

## 🎉 DEFINITION OF DONE

Week 2 is complete when:
- [ ] All 12 tests passing
- [ ] Full universe scan < 65 seconds
- [ ] Performance improvement documented
- [ ] Code reviewed and merged
- [ ] Ready for production deployment

---

**Status:** Week 2 started - Ray workers increased to 50  
**Next Action:** Test with 50 workers and implement async I/O  
**Last Updated:** December 14, 2024
