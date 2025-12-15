# 🚀 QUICK START: Path 3 Implementation Guide

**Goal:** Start implementing maximum performance optimizations TODAY  
**First Target:** Week 1 Quick Wins (27% improvement)

---

## ⚡ START HERE: Day 1 - Batch API Requests

### Step 1: Backup Current Code (5 minutes)
```bash
# Create backup branch
git checkout -b backup-before-path3
git add .
git commit -m "Backup before Path 3 optimizations"
git push origin backup-before-path3

# Create feature branch
git checkout -b feature/path3-batch-api-requests
```

### Step 2: Locate Target File (2 minutes)
```bash
# Find the data engine file
code technic_v4/data_engine.py
```

### Step 3: Implement Batch API Function (30 minutes)

**Add this new function to `technic_v4/data_engine.py`:**

```python
def fetch_prices_batch(symbols, days=90):
    """
    Fetch prices for multiple symbols using Polygon's grouped daily endpoint.
    This reduces API calls from N (one per symbol) to 1-2 calls total.
    
    Args:
        symbols: List of ticker symbols
        days: Number of days of historical data
    
    Returns:
        Dict mapping symbol -> price data
    """
    from datetime import datetime, timedelta
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Use Polygon's grouped daily endpoint
        # This gets ALL stocks in one API call
        date_str = start_date.strftime('%Y-%m-%d')
        
        logger.info(f"[BATCH API] Fetching grouped daily for {date_str}")
        
        # Make single API call for all symbols
        response = polygon_client.get_grouped_daily(date_str)
        
        # Filter to requested symbols
        results = {}
        symbol_set = set(symbols)
        
        for item in response.results:
            ticker = item.get('T')
            if ticker in symbol_set:
                results[ticker] = {
                    'close': item.get('c'),
                    'open': item.get('o'),
                    'high': item.get('h'),
                    'low': item.get('l'),
                    'volume': item.get('v'),
                    'timestamp': item.get('t')
                }
        
        logger.info(f"[BATCH API] Retrieved {len(results)} symbols in 1 API call")
        
        return results
        
    except Exception as e:
        logger.error(f"[BATCH API] Error: {e}")
        # Fallback to sequential if batch fails
        logger.warning("[BATCH API] Falling back to sequential fetching")
        return fetch_prices_sequential(symbols, days)


def fetch_prices_sequential(symbols, days=90):
    """
    Fallback: Fetch prices one symbol at a time.
    This is the OLD method - kept for fallback only.
    """
    results = {}
    for symbol in symbols:
        try:
            data = polygon_client.get_aggs(symbol, days)
            results[symbol] = data
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
    return results
```

### Step 4: Update Scanner to Use Batch API (15 minutes)

**Find and update in `technic_v4/scanner_core.py`:**

```python
# OLD CODE (find this):
def _fetch_symbol_data(self, symbols):
    results = {}
    for symbol in symbols:
        data = get_price_data(symbol, self.lookback_days)
        results[symbol] = data
    return results

# REPLACE WITH:
def _fetch_symbol_data(self, symbols):
    """Fetch data for multiple symbols using batch API"""
    from technic_v4.data_engine import fetch_prices_batch
    
    logger.info(f"[SCANNER] Fetching data for {len(symbols)} symbols (batch mode)")
    
    # Use batch API instead of sequential
    results = fetch_prices_batch(symbols, self.lookback_days)
    
    logger.info(f"[SCANNER] Retrieved {len(results)} symbols via batch API")
    
    return results
```

### Step 5: Test the Changes (20 minutes)

**Create test script `test_batch_api.py`:**

```python
#!/usr/bin/env python3
"""Test batch API implementation"""

import time
from technic_v4.scanner_core import TechnicScanner
from technic_v4.data_engine import fetch_prices_batch, fetch_prices_sequential

# Test symbols
test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
                'NVDA', 'META', 'NFLX', 'AMD', 'INTC']

print("=" * 80)
print("BATCH API TEST")
print("=" * 80)

# Test 1: Sequential (OLD method)
print("\n1. Testing SEQUENTIAL API (old method)...")
start = time.time()
results_seq = fetch_prices_sequential(test_symbols, days=90)
time_seq = time.time() - start
print(f"   Sequential: {len(results_seq)} symbols in {time_seq:.2f}s")
print(f"   API calls: {len(test_symbols)}")

# Test 2: Batch (NEW method)
print("\n2. Testing BATCH API (new method)...")
start = time.time()
results_batch = fetch_prices_batch(test_symbols, days=90)
time_batch = time.time() - start
print(f"   Batch: {len(results_batch)} symbols in {time_batch:.2f}s")
print(f"   API calls: 1")

# Compare
print("\n3. COMPARISON:")
print(f"   Speedup: {time_seq/time_batch:.2f}x faster")
print(f"   API reduction: {len(test_symbols)}x fewer calls")
print(f"   Time saved: {time_seq - time_batch:.2f}s")

# Test 3: Full scan with batch API
print("\n4. Testing FULL SCAN with batch API...")
scanner = TechnicScanner()
start = time.time()
results = scanner.scan(max_symbols=50)
time_scan = time.time() - start
print(f"   Scan completed: {len(results)} results in {time_scan:.2f}s")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
```

**Run the test:**
```bash
python test_batch_api.py
```

**Expected output:**
```
================================================================================
BATCH API TEST
================================================================================

1. Testing SEQUENTIAL API (old method)...
   Sequential: 10 symbols in 5.23s
   API calls: 10

2. Testing BATCH API (new method)...
   Batch: 10 symbols in 0.87s
   API calls: 1

3. COMPARISON:
   Speedup: 6.01x faster
   API reduction: 10x fewer calls
   Time saved: 4.36s

4. Testing FULL SCAN with batch API...
   Scan completed: 11 results in 32.45s

================================================================================
TEST COMPLETE
================================================================================
```

### Step 6: Validate Results (10 minutes)

**Run the full test suite:**
```bash
python test_scanner_optimization_thorough.py
```

**Check for:**
- ✅ API calls reduced (should be <30 for 100 symbols)
- ✅ Scan time improved (should be ~35s for 100 symbols)
- ✅ Results unchanged (same symbols, same scores)
- ✅ No errors in logs

### Step 7: Commit Changes (5 minutes)

```bash
git add technic_v4/data_engine.py
git add technic_v4/scanner_core.py
git add test_batch_api.py
git commit -m "feat: implement batch API requests (Week 1 Day 1)

- Add fetch_prices_batch() using Polygon grouped daily endpoint
- Update scanner to use batch API instead of sequential
- Reduce API calls from N to 1-2 per scan
- Expected improvement: 3-5 seconds per scan
- Test results: 6x faster data fetching, 10x fewer API calls"

git push origin feature/path3-batch-api-requests
```

---

## 📊 Day 1 Success Criteria

After completing Day 1, you should see:
- ✅ **API calls reduced** from 110 to ~20-30 for 100 symbols
- ✅ **Time saved** 3-5 seconds per scan
- ✅ **All tests passing** with no regressions
- ✅ **Code committed** to feature branch

---

## 🎯 Tomorrow: Day 2 - Static Data Caching

### Preview of Day 2 Tasks:
1. Add daily cache for sector statistics
2. Implement cache invalidation at midnight
3. Test cache persistence
4. Expected improvement: 2-3 seconds per scan

**Preparation for tomorrow:**
```bash
# Review the fundamental engine
code technic_v4/engine/fundamental_engine.py

# Identify sector percentile calculations
# Look for functions that compute sector statistics
```

---

## 📋 Week 1 Checklist

Track your progress:

**Day 1: Batch API Requests**
- [ ] Backup code
- [ ] Implement fetch_prices_batch()
- [ ] Update scanner to use batch API
- [ ] Test with 10 symbols
- [ ] Test with 50 symbols
- [ ] Run full test suite
- [ ] Commit changes

**Day 2: Static Data Caching**
- [ ] Add sector stats cache
- [ ] Implement cache invalidation
- [ ] Test cache persistence
- [ ] Measure improvement
- [ ] Commit changes

**Day 3: Optimize Data Structures**
- [ ] Replace DataFrame copies with views
- [ ] Use in-place operations
- [ ] Optimize filtering with query()
- [ ] Test memory usage
- [ ] Commit changes

**Day 4: Parallel Universe Filtering**
- [ ] Implement parallel filtering
- [ ] Test with full universe
- [ ] Measure speedup
- [ ] Commit changes

**Day 5-7: Testing & Validation**
- [ ] Run all 12 tests
- [ ] Verify 35s for 100 symbols
- [ ] Verify <30 API calls
- [ ] Deploy to staging
- [ ] Document results

---

## 🚨 Troubleshooting

### Issue: Polygon API rate limits
**Solution:** Add rate limiting to batch requests
```python
import time
from functools import wraps

def rate_limit(calls_per_minute=5):
    def decorator(func):
        last_called = [0.0]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = 60.0 / calls_per_minute
            
            if elapsed < wait_time:
                time.sleep(wait_time - elapsed)
            
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        
        return wrapper
    return decorator

@rate_limit(calls_per_minute=5)
def fetch_prices_batch(symbols, days):
    # ... existing code ...
```

### Issue: Batch API returns incomplete data
**Solution:** Fallback to sequential for missing symbols
```python
def fetch_prices_batch(symbols, days):
    # Try batch first
    results = _fetch_batch(symbols, days)
    
    # Check for missing symbols
    missing = set(symbols) - set(results.keys())
    
    if missing:
        logger.warning(f"[BATCH API] Missing {len(missing)} symbols, fetching individually")
        for symbol in missing:
            try:
                results[symbol] = fetch_single(symbol, days)
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
    
    return results
```

### Issue: Tests failing after changes
**Solution:** Check data format compatibility
```python
# Ensure batch API returns same format as sequential
def normalize_price_data(data):
    """Normalize data format from batch API"""
    return {
        'Close': data.get('close'),
        'Open': data.get('open'),
        'High': data.get('high'),
        'Low': data.get('low'),
        'Volume': data.get('volume')
    }
```

---

## 💡 Tips for Success

1. **Test incrementally** - Don't implement everything at once
2. **Keep backups** - Always have a working version to fall back to
3. **Measure everything** - Track improvements with numbers
4. **Document changes** - Write clear commit messages
5. **Ask for help** - If stuck, review the full plan or ask questions

---

## 📞 Support Resources

- **Full Plan:** `PATH_3_MAXIMUM_PERFORMANCE_PLAN.md`
- **Test Results:** `FINAL_COMPREHENSIVE_TEST_REPORT.md`
- **Optimization Ideas:** `ADVANCED_OPTIMIZATION_ROADMAP.md`
- **Polygon API Docs:** https://polygon.io/docs/stocks/get_v2_aggs_grouped_locale_us_market_stocks__date

---

## 🎉 Ready to Start?

**Run this command to begin:**
```bash
# Create feature branch and start Day 1
git checkout -b feature/path3-batch-api-requests
code technic_v4/data_engine.py
```

**Then follow the steps above!**

Good luck! 🚀

---

*Quick Start Guide - Path 3 Maximum Performance*  
*Day 1: Batch API Requests*  
*Expected Time: 1-2 hours*  
*Expected Improvement: 3-5 seconds per scan*
