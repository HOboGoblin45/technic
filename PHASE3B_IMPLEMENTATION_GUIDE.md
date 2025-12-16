# Phase 3B Implementation Guide - Week 1
## Complete Vectorization & Ray Optimization

**Goal:** Achieve 3-6 minute scans (2-3x speedup)
**Timeline:** 5 days
**Cost:** $85/month (Render Pro Plus - no change)

---

## Day 1-2: Complete Batch Processor Integration

### Step 1: Update `technic_v4/engine/batch_processor.py`

Add the complete batch processing method:

```python
def process_batch(self, symbols_data: Dict[str, pd.DataFrame], config) -> List[Dict]:
    """
    Process multiple symbols using vectorized operations
    
    Args:
        symbols_data: Dict of {symbol: price_dataframe}
        config: ScanConfig object
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    # Convert to list for batch processing
    symbols = list(symbols_data.keys())
    dataframes = list(symbols_data.values())
    
    # Filter out None/empty dataframes
    valid_data = [(sym, df) for sym, df in zip(symbols, dataframes) 
                  if df is not None and not df.empty and len(df) >= 20]
    
    if not valid_data:
        return results
    
    symbols, dataframes = zip(*valid_data)
    
    # Batch compute indicators for all symbols
    logger.info(f"[BATCH] Processing {len(symbols)} symbols")
    
    for symbol, df in zip(symbols, dataframes):
        try:
            # Compute indicators (vectorized internally)
            indicators = self.compute_indicators_single(df)
            
            # Create result dict
            result = {
                'Symbol': symbol,
                'Close': float(df['Close'].iloc[-1]),
                'Volume': float(df['Volume'].iloc[-1]),
                **indicators
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"[BATCH] Error processing {symbol}: {e}")
            continue
    
    logger.info(f"[BATCH] Completed {len(results)} symbols")
    return results
```

### Step 2: Update `technic_v4/scanner_core.py`

Integrate batch processor into main scan flow:

```python
def _run_symbol_scans(
    config: ScanConfig,
    universe: List[UniverseRow],
    regime_tags: Optional[dict],
    effective_lookback: int,
    settings=None,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
    price_cache: Optional[Dict[str, pd.DataFrame]] = None,  # NEW
) -> Tuple[pd.DataFrame, dict]:
    """
    Execute per-symbol scans with batch processing
    """
    from technic_v4.engine.batch_processor import BatchProcessor
    
    rows: List[pd.Series] = []
    attempted = kept = errors = rejected = 0
    total_symbols = len(universe)
    
    if settings is None:
        settings = get_settings()
    
    # Use batch processing if price_cache provided
    if price_cache:
        logger.info("[BATCH] Using batch processing mode")
        
        processor = BatchProcessor()
        
        # Process in chunks of 500
        chunk_size = 500
        for i in range(0, len(universe), chunk_size):
            chunk = universe[i:i+chunk_size]
            chunk_data = {
                urow.symbol: price_cache.get(urow.symbol)
                for urow in chunk
            }
            
            # Batch process chunk
            chunk_results = processor.process_batch(chunk_data, config)
            
            # Convert to Series and add metadata
            for result in chunk_results:
                row = pd.Series(result)
                # Add sector/industry from universe
                urow = next((u for u in chunk if u.symbol == result['Symbol']), None)
                if urow:
                    row['Sector'] = urow.sector or ""
                    row['Industry'] = urow.industry or ""
                    row['SubIndustry'] = urow.subindustry or ""
                
                rows.append(row)
                kept += 1
            
            attempted += len(chunk)
            
            # Progress callback
            if progress_cb:
                progress_cb("Batch processing", i + len(chunk), total_symbols)
        
        stats = {
            "attempted": attempted,
            "kept": kept,
            "errors": 0,
            "rejected": attempted - kept,
        }
        
        return pd.DataFrame(rows), stats
    
    # Fallback to original Ray processing if no cache
    else:
        # ... existing Ray processing code ...
        pass
```

### Step 3: Test Batch Processing

Create test script `test_batch_processing.py`:

```python
#!/usr/bin/env python3
"""Test batch processing implementation"""

import time
from technic_v4.scanner_core import ScanConfig, run_scan
from technic_v4 import data_engine

def test_batch_processing():
    """Test batch processing with 100 symbols"""
    
    print("=" * 80)
    print("BATCH PROCESSING TEST")
    print("=" * 80)
    
    # Create config for 100 symbols
    config = ScanConfig(
        max_symbols=100,
        lookback_days=150,
        sectors=["Technology", "Healthcare"]
    )
    
    # Run scan
    start = time.time()
    results, status = run_scan(config)
    elapsed = time.time() - start
    
    # Verify results
    print(f"\n✅ Scan completed in {elapsed:.2f} seconds")
    print(f"✅ Results: {len(results)} symbols")
    print(f"✅ Per-symbol: {elapsed/len(results):.3f}s")
    
    # Check for required columns
    required_cols = ['Symbol', 'TechRating', 'Signal', 'Close', 'Volume']
    missing = [col for col in required_cols if col not in results.columns]
    
    if missing:
        print(f"❌ Missing columns: {missing}")
        return False
    
    print(f"✅ All required columns present")
    
    # Target: <60 seconds for 100 symbols (0.6s per symbol)
    target_time = 60
    if elapsed <= target_time:
        print(f"✅ PASSED: {elapsed:.2f}s <= {target_time}s target")
        return True
    else:
        print(f"⚠️  SLOW: {elapsed:.2f}s > {target_time}s target")
        return False

if __name__ == "__main__":
    success = test_batch_processing()
    exit(0 if success else 1)
```

Run test:
```bash
python test_batch_processing.py
```

**Expected Result:** 100 symbols in <60 seconds

---

## Day 3-4: Optimize Ray Workers

### Step 1: Update `technic_v4/engine/alpha_inference.py`

Add global model cache and batch inference:

```python
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# Global model cache (loaded once, reused forever)
_MODEL_CACHE: Dict[str, any] = {}

def get_model_cached(model_name: str):
    """Load model once and cache globally"""
    if model_name not in _MODEL_CACHE:
        logger.info(f"[MODEL CACHE] Loading {model_name}")
        _MODEL_CACHE[model_name] = load_xgb_model(model_name)
    return _MODEL_CACHE[model_name]

def score_alpha_batch(df_batch: pd.DataFrame) -> pd.Series:
    """
    Batch ML inference for multiple symbols
    
    Args:
        df_batch: DataFrame with features for multiple symbols
        
    Returns:
        Series of predictions indexed by symbol
    """
    if df_batch.empty:
        return pd.Series(dtype=float)
    
    # Get cached model
    model = get_model_cached('alpha_5d')
    
    # Extract features for all symbols at once
    feature_cols = [
        'RSI', 'MACD', 'MACD_signal', 'BB_width', 'ATR_pct',
        'Volume_ratio', 'mom_21', 'mom_63'
    ]
    
    # Stack features into 2D array
    all_features = df_batch[feature_cols].values
    
    # Single vectorized prediction
    predictions = model.predict(all_features)
    
    # Return as Series
    return pd.Series(predictions, index=df_batch.index)

def score_alpha_10d_batch(df_batch: pd.DataFrame) -> pd.Series:
    """Batch inference for 10d model"""
    if df_batch.empty:
        return pd.Series(dtype=float)
    
    model = get_model_cached('alpha_10d')
    
    feature_cols = [
        'RSI', 'MACD', 'MACD_signal', 'BB_width', 'ATR_pct',
        'Volume_ratio', 'mom_21', 'mom_63', 'mom_126'
    ]
    
    all_features = df_batch[feature_cols].values
    predictions = model.predict(all_features)
    
    return pd.Series(predictions, index=df_batch.index)
```

### Step 2: Update `technic_v4/engine/ray_runner.py`

Implement stateful Ray workers with model caching:

```python
import ray
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Initialize Ray with more workers for I/O-bound tasks
ray.init(num_cpus=200, ignore_reinit_error=True)

@ray.remote
class BatchWorker:
    """
    Stateful Ray worker that caches ML models
    Avoids repeated model loading overhead
    """
    
    def __init__(self):
        self.models = {}
        self.worker_id = None
        self._load_models()
    
    def _load_models(self):
        """Load all ML models once per worker"""
        from technic_v4.engine.alpha_inference import load_xgb_model
        
        logger.info("[WORKER] Loading models")
        self.models['alpha_5d'] = load_xgb_model('alpha_5d')
        self.models['alpha_10d'] = load_xgb_model('alpha_10d')
        logger.info("[WORKER] Models loaded")
    
    def process_batch(self, symbols_data: Dict[str, pd.DataFrame], config) -> List[Dict]:
        """
        Process a batch of symbols with cached models
        
        Args:
            symbols_data: Dict of {symbol: price_dataframe}
            config: ScanConfig
            
        Returns:
            List of result dicts
        """
        from technic_v4.engine.batch_processor import BatchProcessor
        
        processor = BatchProcessor()
        results = processor.process_batch(symbols_data, config)
        
        # Add ML predictions using cached models
        for result in results:
            try:
                # Use cached models for predictions
                features = self._extract_features(result)
                result['Alpha5d'] = self.models['alpha_5d'].predict([features])[0]
                result['Alpha10d'] = self.models['alpha_10d'].predict([features])[0]
            except Exception as e:
                logger.error(f"[WORKER] ML prediction error: {e}")
                result['Alpha5d'] = 0.0
                result['Alpha10d'] = 0.0
        
        return results
    
    def _extract_features(self, result: Dict) -> List[float]:
        """Extract features for ML model"""
        return [
            result.get('RSI', 50),
            result.get('MACD', 0),
            result.get('MACD_signal', 0),
            result.get('BB_width', 0),
            result.get('ATR_pct', 0.02),
            result.get('Volume_ratio', 1.0),
            result.get('mom_21', 0),
            result.get('mom_63', 0),
        ]

# Create worker pool (20 workers, each with cached models)
_WORKER_POOL = [BatchWorker.remote() for _ in range(20)]

def run_ray_scans_optimized(
    symbols: List[str],
    config,
    regime_tags: Optional[dict],
    price_cache: Dict[str, pd.DataFrame]
) -> List[Dict]:
    """
    Run scans using optimized Ray workers with model caching
    
    Args:
        symbols: List of symbols to scan
        config: ScanConfig
        regime_tags: Market regime info
        price_cache: Pre-fetched price data
        
    Returns:
        List of scan results
    """
    # Split symbols into batches for workers
    num_workers = len(_WORKER_POOL)
    batch_size = len(symbols) // num_workers + 1
    
    batches = []
    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i+batch_size]
        batch_data = {sym: price_cache.get(sym) for sym in batch_symbols}
        batches.append(batch_data)
    
    # Distribute work to workers
    futures = [
        _WORKER_POOL[i % num_workers].process_batch.remote(batch, config)
        for i, batch in enumerate(batches)
    ]
    
    # Gather results
    all_results = []
    for future in futures:
        batch_results = ray.get(future)
        all_results.extend(batch_results)
    
    return all_results
```

### Step 3: Update `technic_v4/config/settings.py`

Increase max_workers:

```python
@dataclass
class Settings:
    # ... existing fields ...
    
    # PHASE 3B: Increased for I/O-bound tasks
    max_workers: int = field(default=200)  # Was 50
    
    # Enable Ray optimization
    use_ray: bool = field(default=True)
```

### Step 4: Test Ray Optimization

Create test script `test_ray_optimization.py`:

```python
#!/usr/bin/env python3
"""Test Ray worker optimization"""

import time
from technic_v4.scanner_core import ScanConfig, run_scan

def test_ray_workers():
    """Test with 500 symbols"""
    
    print("=" * 80)
    print("RAY WORKER OPTIMIZATION TEST")
    print("=" * 80)
    
    config = ScanConfig(
        max_symbols=500,
        lookback_days=150
    )
    
    start = time.time()
    results, status = run_scan(config)
    elapsed = time.time() - start
    
    print(f"\n✅ Scan completed in {elapsed:.2f} seconds")
    print(f"✅ Results: {len(results)} symbols")
    print(f"✅ Per-symbol: {elapsed/len(results):.3f}s")
    
    # Target: <3 minutes for 500 symbols
    target_time = 180
    if elapsed <= target_time:
        print(f"✅ PASSED: {elapsed:.2f}s <= {target_time}s target")
        return True
    else:
        print(f"⚠️  SLOW: {elapsed:.2f}s > {target_time}s target")
        return False

if __name__ == "__main__":
    success = test_ray_workers()
    exit(0 if success else 1)
```

**Expected Result:** 500 symbols in <3 minutes

---

## Day 5: Testing & Validation

### Comprehensive Test Suite

Create `test_phase3b_complete.py`:

```python
#!/usr/bin/env python3
"""Comprehensive Phase 3B testing"""

import time
import pandas as pd
from technic_v4.scanner_core import ScanConfig, run_scan

def test_small_scan():
    """Test 100 symbols"""
    print("\n" + "=" * 80)
    print("TEST 1: Small Scan (100 symbols)")
    print("=" * 80)
    
    config = ScanConfig(max_symbols=100, lookback_days=150)
    start = time.time()
    results, _ = run_scan(config)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Results: {len(results)}")
    print(f"Per-symbol: {elapsed/len(results):.3f}s")
    
    assert elapsed < 60, f"Too slow: {elapsed:.2f}s > 60s"
    assert len(results) > 0, "No results"
    print("✅ PASSED")
    return True

def test_medium_scan():
    """Test 1000 symbols"""
    print("\n" + "=" * 80)
    print("TEST 2: Medium Scan (1000 symbols)")
    print("=" * 80)
    
    config = ScanConfig(max_symbols=1000, lookback_days=150)
    start = time.time()
    results, _ = run_scan(config)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Results: {len(results)}")
    print(f"Per-symbol: {elapsed/len(results):.3f}s")
    
    # Target: <5 minutes
    assert elapsed < 300, f"Too slow: {elapsed:.2f}s > 300s"
    print("✅ PASSED")
    return True

def test_full_scan():
    """Test 5000 symbols"""
    print("\n" + "=" * 80)
    print("TEST 3: Full Scan (5000 symbols)")
    print("=" * 80)
    
    config = ScanConfig(max_symbols=5000, lookback_days=150)
    start = time.time()
    results, _ = run_scan(config)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s ({elapsed/60:.1f} minutes)")
    print(f"Results: {len(results)}")
    print(f"Per-symbol: {elapsed/len(results):.3f}s")
    
    # Target: 3-6 minutes
    assert elapsed < 360, f"Too slow: {elapsed:.2f}s > 360s (6 min)"
    print("✅ PASSED - Phase 3B Target Achieved!")
    return True

def test_result_quality():
    """Verify result quality"""
    print("\n" + "=" * 80)
    print("TEST 4: Result Quality")
    print("=" * 80)
    
    config = ScanConfig(max_symbols=50, lookback_days=150)
    results, _ = run_scan(config)
    
    # Check required columns
    required = ['Symbol', 'TechRating', 'Signal', 'Close', 'Volume', 
                'RSI', 'MACD', 'ATR_pct']
    missing = [col for col in required if col not in results.columns]
    
    assert not missing, f"Missing columns: {missing}"
    
    # Check for NaN values in critical columns
    for col in ['TechRating', 'Close']:
        nan_count = results[col].isna().sum()
        assert nan_count == 0, f"{col} has {nan_count} NaN values"
    
    print(f"✅ All {len(required)} required columns present")
    print(f"✅ No NaN values in critical columns")
    print("✅ PASSED")
    return True

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("PHASE 3B COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Small Scan", test_small_scan),
        ("Medium Scan", test_medium_scan),
        ("Result Quality", test_result_quality),
        ("Full Scan", test_full_scan),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("=" * 80)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED - PHASE 3B COMPLETE!")
        print("Ready to proceed to Phase 3C (Redis Caching)")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
```

Run complete test suite:
```bash
python test_phase3b_complete.py
```

---

## Success Criteria

✅ **Performance:**
- 100 symbols: <60 seconds
- 1,000 symbols: <5 minutes
- 5,000 symbols: <6 minutes (3-6 minute target)

✅ **Quality:**
- All required columns present
- No NaN values in critical fields
- Results match original implementation

✅ **Stability:**
- No memory leaks
- No crashes or errors
- Graceful error handling

---

## Deployment

Once all tests pass:

```bash
# Commit changes
git add -A
git commit -m "Phase 3B: Vectorization & Ray optimization complete

- Implemented batch processor integration
- Added global model caching
- Optimized Ray workers (200 workers)
- Achieved 2-3x speedup (3-6 minute scans)
- All tests passing"

# Push to Render
git push origin main
```

Monitor Render deployment and verify performance in production.

---

## Next Steps

After Phase 3B is complete and deployed:
- **Week 2:** Proceed to Phase 3C (Redis Caching)
- Target: 1.5-3 minute scans (2x additional speedup)
