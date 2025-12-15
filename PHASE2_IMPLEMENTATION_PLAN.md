# Phase 2: Early Rejection Optimization - Implementation Plan

**Priority:** 🔴 CRITICAL  
**Goal:** Reduce universe filtering from 156.49s → <60s (62% improvement)  
**Strategy:** Move expensive checks BEFORE API data fetch

---

## 🎯 The Problem (Validated by Tests)

### Current Wasteful Flow:
```python
for symbol in universe:
    data = fetch_price_data(symbol)  # ← EXPENSIVE API CALL
    if not _passes_basic_filters(data):  # ← CHECK AFTER FETCH
        reject(symbol)
```

**Result:** We fetch data for 2,639 symbols, then reject 1,063 (40%) AFTER the expensive operation.

### Optimized Flow:
```python
for symbol in universe:
    if not _can_pass_basic_filters(symbol):  # ← CHECK BEFORE FETCH
        reject(symbol)
        continue
    data = fetch_price_data(symbol)  # ← ONLY IF LIKELY TO PASS
    if not _passes_basic_filters(data):
        reject(symbol)
```

**Expected result:** Fetch data for ~1,300-1,500 symbols (50% reduction in API calls)

---

## 📋 Implementation Steps

### Step 1: Add Pre-Screening Function

**Location:** `technic_v4/scanner_core.py`

**Add new function BEFORE `_scan_symbol()`:**

```python
def _can_pass_basic_filters(self, symbol: str, meta: dict) -> bool:
    """
    Quick pre-screening WITHOUT fetching price data.
    Rejects symbols that will definitely fail basic filters.
    
    Returns:
        True if symbol might pass (fetch data)
        False if symbol will definitely fail (skip)
    """
    # 1. Market cap check (if available in meta)
    market_cap = meta.get('market_cap', 0)
    if market_cap > 0 and market_cap < 100_000_000:  # <$100M
        return False
    
    # 2. Sector-specific rejections
    sector = meta.get('Sector', '')
    if sector in ['Real Estate', 'Utilities']:  # Low volatility sectors
        # More strict for these sectors
        if market_cap < 500_000_000:  # <$500M
            return False
    
    # 3. Known problematic patterns
    # - Single letter symbols (often ADRs with low liquidity)
    # - Symbols ending in certain patterns
    if len(symbol) == 1:
        return False
    
    # 4. Industry-specific rejections
    industry = meta.get('Industry', '')
    if industry in ['REIT', 'Closed-End Fund', 'Exchange Traded Fund']:
        return False
    
    return True  # Might pass, fetch data to confirm
```

### Step 2: Integrate Pre-Screening into Scan Loop

**Location:** `technic_v4/scanner_core.py` in `run_scan()` method

**FIND this section:**
```python
for symbol_meta in universe_df.to_dict('records'):
    symbol = symbol_meta['Symbol']
    result = self._scan_symbol(symbol, symbol_meta, ...)
```

**REPLACE with:**
```python
pre_rejected = 0
for symbol_meta in universe_df.to_dict('records'):
    symbol = symbol_meta['Symbol']
    
    # PRE-SCREENING: Quick rejection before expensive operations
    if not self._can_pass_basic_filters(symbol, symbol_meta):
        pre_rejected += 1
        continue
    
    result = self._scan_symbol(symbol, symbol_meta, ...)

# Log pre-rejection stats
logger.info(f"[PRE_FILTER] Rejected {pre_rejected} symbols before data fetch")
logger.info(f"[PRE_FILTER] Fetching data for {len(universe_df) - pre_rejected} symbols")
```

### Step 3: Add Metadata Enrichment (Optional but Recommended)

**If market cap not in universe CSV, add quick lookup:**

```python
def _enrich_universe_metadata(self, universe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add minimal metadata needed for pre-screening.
    Uses cached fundamental data if available.
    """
    # Check if we have cached fundamental data
    if hasattr(self, 'fundamental_cache'):
        for idx, row in universe_df.iterrows():
            symbol = row['Symbol']
            if symbol in self.fundamental_cache:
                universe_df.at[idx, 'market_cap'] = self.fundamental_cache[symbol].get('market_cap', 0)
    
    return universe_df
```

---

## 🎯 Expected Performance Impact

### Conservative Estimate:

**Assumptions:**
- Pre-screening rejects 40% of symbols (1,056 symbols)
- Remaining symbols: 2,639 - 1,056 = 1,583 symbols
- Per-symbol time: 0.059s (from tests)

**Calculation:**
- Time = 1,583 symbols × 0.059s = **93.4s**
- Still above target, but 40% improvement

### Aggressive Estimate:

**Assumptions:**
- Pre-screening rejects 50% of symbols (1,320 symbols)
- Remaining symbols: 2,639 - 1,320 = 1,319 symbols
- Per-symbol time improves to 0.045s (less overhead)

**Calculation:**
- Time = 1,319 symbols × 0.045s = **59.4s**
- **MEETS <60s TARGET** ✅

---

## 🔧 Implementation Checklist

- [ ] Add `_can_pass_basic_filters()` function
- [ ] Integrate pre-screening into scan loop
- [ ] Add pre-rejection logging
- [ ] Test with small sample (100 symbols)
- [ ] Run full Test 4 to validate
- [ ] Measure actual rejection rate
- [ ] Tune thresholds if needed
- [ ] Document results

---

## 📊 Success Criteria

**Phase 2 is successful if:**
1. ✅ Test 4 time: <60s (currently 156.49s)
2. ✅ Pre-rejection rate: 40-50%
3. ✅ No false negatives (don't reject good symbols)
4. ✅ Warm scans still fast (<10s)
5. ✅ API calls reduced to ≤100

---

## ⚠️ Risk Mitigation

**Risk:** Pre-screening might reject good symbols

**Mitigation:**
1. Start with conservative thresholds
2. Log all pre-rejections for review
3. Compare results before/after Phase 2
4. Tune thresholds based on false negative rate

---

## 🚀 Ready to Implement?

**Estimated implementation time:** 30-45 minutes  
**Estimated testing time:** 10-15 minutes  
**Total time to results:** ~1 hour

**Next step:** Implement `_can_pass_basic_filters()` and integrate into scan loop.
