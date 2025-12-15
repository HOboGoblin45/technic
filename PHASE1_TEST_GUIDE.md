# Phase 1 Testing Guide

## How to Run the Test

```bash
python test_scanner_optimization_thorough.py
```

---

## What to Look For in Test Output

### 1. Universe Filtering (Test 4)

**Key Metrics:**
```
Test 4: Universe Filtering Performance
  Time: ??? seconds (target: <60s)
  Symbols processed: ???
  Per-symbol time: ??? seconds
```

**Expected Results:**
- ✅ **Good:** 80-90s (40-45% improvement from 152.15s)
- ✅ **Great:** 60-80s (47-60% improvement)
- ✅ **Excellent:** <60s (>60% improvement - meets target!)

---

### 2. Smart Filter Log Messages

Look for these log entries in the output:

```
[SMART_FILTER] Removed X symbols with invalid tickers
[SMART_FILTER] Focused on liquid sectors, removed X symbols
[SMART_FILTER] Removed X leveraged/volatility products
[SMART_FILTER] Removed X potential penny stocks
[SMART_FILTER] Sector distribution: Technology=XXX, Healthcare=XXX, ...
[SMART_FILTER] Reduced universe: 5277 → XXXX symbols (XX.X% reduction)
```

**What to Check:**
- Total symbols after smart filtering (expect ~2,400-2,600 vs 2,648 before)
- Leveraged products removed (expect ~18-24 symbols)
- Sector distribution (should show top 5 sectors)

---

### 3. Universe Size Comparison

**Before Phase 1:**
```
[UNIVERSE] loaded 5277 symbols from ticker_universe.csv
[SMART_FILTER] Reduced universe: 5277 → 2648 symbols (49.8% reduction)
```

**After Phase 1 (Expected):**
```
[UNIVERSE] loaded 5277 symbols from ticker_universe.csv
[SMART_FILTER] Removed 18 leveraged/volatility products
[SMART_FILTER] Removed 5 potential penny stocks
[SMART_FILTER] Sector distribution: Technology=450, Healthcare=380, ...
[SMART_FILTER] Reduced universe: 5277 → 2625 symbols (50.2% reduction)
```

**Analysis:**
- Small reduction expected (2648 → 2625 = 23 symbols)
- This is because most filtering happens in `_passes_basic_filters()` AFTER fetching data
- **Phase 2 will move those checks BEFORE fetching** for bigger gains

---

### 4. Performance Breakdown

**Current Bottleneck (from analysis):**
```
Total time: 152.15s for 2,648 symbols
Per-symbol: 0.057s/symbol

Breakdown:
- Smart filtering: ~1s (negligible)
- Per-symbol processing: 151s
  - Fetch price data: ~0.020s/symbol (53s total)
  - _passes_basic_filters: ~0.015s/symbol (40s total) ← WASTEFUL
  - Scoring/indicators: ~0.022s/symbol (58s total)
```

**Phase 1 Impact:**
- Reduces symbols from 2,648 → ~2,625 (minimal)
- Time savings: ~1-2 seconds (not significant)
- **This is expected!** Phase 1 is about setting up for Phase 2

**Phase 2 Will:**
- Move `_passes_basic_filters()` checks BEFORE fetch
- Reject ~1,000 symbols early (no fetch needed)
- Save: 1,000 × 0.020s = 20 seconds
- Expected result: 152s → 60-80s

---

## Decision Tree After Testing

### If Test 4 shows 140-152s (minimal improvement):
✅ **This is EXPECTED and OK!**
- Phase 1 was about infrastructure
- Proceed immediately to Phase 2 (early rejection)
- Phase 2 will deliver the big wins

### If Test 4 shows 80-100s (moderate improvement):
✅ **Better than expected!**
- Some symbols were rejected by new filters
- Still proceed to Phase 2 for final push to <60s

### If Test 4 shows <80s (major improvement):
🎉 **Excellent!**
- Phase 1 alone nearly met target
- Phase 2 will easily get us to <60s

### If Test 4 shows >152s (regression):
⚠️ **Investigation needed**
- Check for errors in smart filter logic
- Review log output for issues
- May need to adjust filter criteria

---

## Next Steps After Testing

1. **Share test results** (especially Test 4 time and log output)
2. **I'll analyze** the actual vs expected performance
3. **Proceed to Phase 2** (early rejection optimization)
4. **Retest** to validate final performance

---

## Quick Commands

**Run full test suite:**
```bash
python test_scanner_optimization_thorough.py
```

**Run only Test 4 (universe filtering):**
```bash
python test_scanner_optimization_thorough.py --test 4
```

**Check log output:**
```bash
# Look for [SMART_FILTER] messages in console output
```

---

**Status:** Waiting for test results  
**Expected:** Minimal improvement (Phase 1 is setup for Phase 2)  
**Target:** Phase 2 will deliver 152s → <60s
