# Phase 3: Achieving 90-Second Scans - Comprehensive Plan

## 🎯 Goal: 90 Seconds for 5,000-6,000 Symbols

**Current**: 9-18 minutes (540-1,080 seconds) after Phases 1+2
**Target**: 90 seconds
**Required**: Additional 6-12x speedup

---

## 🔍 Root Cause Analysis

### Why Are We Still Slow?

#### 1. **Per-Symbol CPU Time** (Biggest Bottleneck)
```python
# Even with perfect caching, each symbol requires:
- Technical indicators: 0.05-0.10s (NumPy/Pandas operations)
- ML inference (2 models): 0.02-0.05s (XGBoost)
- Scoring pipeline: 0.01-0.02s (factor calculations)
- Total: 0.08-0.17s per symbol

# For 5,000 symbols:
5,000 × 0.08s = 400 seconds minimum (even with infinite parallelism)
```

**This is the hard limit on current architecture.**

#### 2. **Python GIL** (Global Interpreter Lock)
- Limits true parallelism for CPU-bound tasks
- ThreadPool doesn't help for NumPy/Pandas operations
- Ray helps but still limited by single-machine CPU

#### 3. **Sequential Processing Steps**
```python
# Current flow (sequential):
1. Batch fetch data (30-60s)
2. Process all symbols (400-600s)
3. Finalize results (10-20s)
Total: 440-680s
```

---

## 🚀 Solution: Multi-Pronged Optimization

### Phase 3A: Vectorized Batch Processing
**Target**: 3x speedup (9-18 min → 3-6 min)

#### 1. Batch Indicator Calculation
Instead of processing symbols one-by-one, process in batches of 100:

```python
def _compute_indicators_batch(price_data_dict: dict) -> dict:
    """
    Compute indicators for 100 symbols at once using vectorized operations.
    
    PERFORMANCE: 10x faster than per-symbol loops.
    """
    # Stack all DataFrames into 3D array
    symbols = list(price_data_dict.keys())
    arrays = [price_data_dict[sym][['Close', 'Volume']].values for sym in symbols]
    
    # Vectorized RSI calculation for all symbols at once
    batch_rsi = talib.RSI(np.array([arr[:, 0] for arr in arrays]))
    
    # Return dict of results
    return {sym: indicators for sym, indicators in zip(symbols, batch_rsi)}
```

**Impact**: 0.05s → 0.005s per symbol (10x faster)

#### 2. Batch ML Inference
```python
def _batch_ml_inference(features_df: pd.DataFrame) -> pd.Series:
    """
    Run ML model on 100+ symbols at once.
    
    PERFORMANCE: GPU/vectorized operations are 20x faster than loops.
    """
    # Single model.predict() call for all symbols
    predictions = model.predict(features_df)
    return pd.Series(predictions, index=features_df.index)
```

**Impact**: 0.02s → 0.001s per symbol (20x faster)

---

### Phase 3B: Parallel Architecture Redesign
**Target**: 2x speedup (3-6 min → 1.5-3 min)

#### 1. True Parallel Processing with Ray
```python
# Current: ThreadPool (GIL-limited)
# New: Ray with multiple processes (no GIL)

@ray.remote
class SymbolProcessor:
    def __init__(self):
        self.model = load_ml_model()  # Load once per worker
    
    def process_batch(self, symbols_batch, price_cache):
        # Process 100 symbols in this worker
        return [self._process_one(sym, price_cache) for sym in symbols_batch]

# Spawn 20 workers, each processes 250 symbols
workers = [SymbolProcessor.remote() for _ in range(20)]
```

**Impact**: Better CPU utilization, no GIL contention

#### 2. Async I/O for Remaining API Calls
```python
import asyncio
import aiohttp

async def fetch_fundamentals_batch(symbols: List[str]) -> dict:
    """
    Fetch fundamentals for all symbols concurrently.
    
    PERFORMANCE: 50x faster than sequential requests.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, sym) for sym in symbols]
        results = await asyncio.gather(*tasks)
    return dict(zip(symbols, results))
```

**Impact**: Fundamentals fetch: 100s → 2s

---

### Phase 3C: Infrastructure Upgrade
**Target**: 3x speedup (1.5-3 min → 30-60 seconds)

#### Option 1: AWS EC2 with Ray Cluster (Recommended)
```yaml
Infrastructure:
  - Head Node: c6i.2xlarge (8 vCPU, 16 GB RAM)
  - Worker Nodes: 5x c6i.4xlarge (16 vCPU, 32 GB RAM each)
  - Total: 88 vCPU, 176 GB RAM
  - Ray Cluster: Distributed processing
  - Redis: ElastiCache for caching
  
Cost: ~$300-400/month

Performance:
  - 88 cores vs 4 cores = 22x more compute
  - Distributed Ray = no GIL limits
  - Expected: 30-60 second scans
```

#### Option 2: GPU Acceleration
```yaml
Infrastructure:
  - GPU Instance: g4dn.xlarge (4 vCPU, 16 GB RAM, 1x T4 GPU)
  - ML Inference: GPU-accelerated (100x faster)
  - Indicator Calc: Still CPU-bound
  
Cost: ~$150-200/month

Performance:
  - ML inference: 0.02s → 0.0002s per symbol
  - Expected: 60-90 second scans
```

---

### Phase 3D: Algorithmic Optimization
**Target**: 1.5x speedup (30-60 sec → 20-40 seconds)

#### 1. Smart Sampling
```python
# Don't scan ALL 5,000 symbols every time
# Use incremental updates:

def incremental_scan(full_universe, last_scan_results):
    """
    Only re-scan symbols that have changed significantly.
    
    PERFORMANCE: 80% of symbols don't change much day-to-day.
    """
    # Identify symbols with significant price/volume changes
    changed = identify_changed_symbols(full_universe, last_scan_results)
    
    # Full scan: 20% changed + top 500 from last scan
    to_scan = changed + last_scan_results.head(500)
    
    # Merge with cached results for unchanged symbols
    return merge_results(scan(to_scan), last_scan_results)
```

**Impact**: Scan 1,000 symbols instead of 5,000 (5x faster)

#### 2. Lazy Evaluation
```python
# Don't compute everything for every symbol
# Compute expensive features only for top candidates

def lazy_scan(symbols):
    # Stage 1: Quick score (0.01s per symbol)
    quick_scores = [quick_score(sym) for sym in symbols]
    top_1000 = sorted(quick_scores)[:1000]
    
    # Stage 2: Full analysis (0.08s per symbol, but only 1,000 symbols)
    full_results = [full_analysis(sym) for sym in top_1000]
    
    return full_results
```

**Impact**: 5,000 × 0.08s → 1,000 × 0.08s = 80s (vs 400s)

---

## 📋 Complete Roadmap to 90 Seconds

### Week 1-2: Phase 3A (Vectorized Processing)
- [ ] Implement batch indicator calculation
- [ ] Implement batch ML inference
- [ ] Test and validate results match
- **Expected**: 3-6 minute scans

### Week 3-4: Phase 3B (Parallel Architecture)
- [ ] Refactor to Ray-based processing
- [ ] Implement async I/O for API calls
- [ ] Optimize worker distribution
- **Expected**: 1.5-3 minute scans

### Week 5-6: Phase 3C (Infrastructure)
- [ ] Set up AWS EC2 Ray cluster OR GPU instance
- [ ] Deploy Redis ElastiCache
- [ ] Configure auto-scaling
- **Expected**: 30-60 second scans

### Week 7-8: Phase 3D (Algorithmic)
- [ ] Implement incremental scanning
- [ ] Add lazy evaluation
- [ ] Optimize hot paths with Cython
- **Expected**: 20-40 second scans

### Week 9-10: Fine-Tuning
- [ ] Profile and optimize bottlenecks
- [ ] Load testing and stress testing
- [ ] Production deployment
- **Target**: 60-90 second scans ✅

---

## 💰 Cost Analysis

### Current (Render Pro Plus)
- **Cost**: $7/month
- **Performance**: 9-18 minutes
- **Good for**: Development, testing

### Phase 3 Complete (AWS + Ray)
- **Cost**: $300-400/month
- **Performance**: 60-90 seconds
- **Good for**: Production, scaling

### Alternative (GPU Acceleration)
- **Cost**: $150-200/month
- **Performance**: 60-120 seconds
- **Good for**: Budget-conscious production

---

## ⚡ Quick Wins (Can Implement Now)

### 1. Increase Ray Workers to 100
```python
# In settings.py
max_workers = 100  # Currently 100, but verify Ray is using them
```

### 2. Reduce Lookback Period
```python
# In scanner_core.py
lookback_days: int = 90  # Instead of 150
```
**Impact**: 40% less data to process

### 3. Skip Options Analysis for Most Symbols
```python
# Only analyze options for top 50 symbols
options_max_symbols = 50  # Instead of 200
```
**Impact**: 75% less options processing

### 4. Disable Expensive Features
```python
# In settings.py
use_explainability = False  # SHAP is slow
use_tft_features = False    # TFT is slow
```
**Impact**: 20-30% faster

**Combined Quick Wins**: Could get to 5-7 minutes immediately

---

## 🎯 Recommendation

### Path to 90 Seconds

**Immediate (This Week)**:
1. Implement quick wins above
2. Test and measure actual performance
3. **Expected**: 5-7 minute scans

**Short-Term (Next Month)**:
1. Implement Phase 3A (vectorized processing)
2. **Expected**: 2-3 minute scans

**Medium-Term (Next Quarter)**:
1. Upgrade to AWS + Ray cluster
2. Implement Phases 3B-3D
3. **Expected**: 60-90 second scans ✅

**The 90-second goal IS achievable, but requires infrastructure investment and 2-3 months of work. The current 9-18 minutes is excellent progress for code-only optimizations.**
