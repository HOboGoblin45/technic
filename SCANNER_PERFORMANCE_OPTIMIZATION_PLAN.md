# 🚀 Scanner Performance Optimization & Refinement Plan

## Objective

Optimize the Technic scanner to run blazingly fast while maintaining accuracy and expanding the development roadmap for a perfect, production-ready app.

---

## 🎯 Current Performance Analysis

### Current Scanner Architecture

**File**: `technic_v4/scanner_core.py`

**Current Flow:**
1. Load universe (~5,000 symbols)
2. Filter by sector/industry
3. Fetch price history for each symbol (sequential or parallel)
4. Compute indicators (TA-Lib)
5. Compute factors (momentum, value, quality, etc.)
6. Build ICS (Institutional Core Score)
7. Compute MERIT Score
8. Rank and sort
9. Return top N results

**Performance Bottlenecks:**
1. **Data Fetching**: Polygon API calls for 5,000+ symbols
2. **Indicator Computation**: TA-Lib calculations per symbol
3. **Sequential Processing**: Some operations not parallelized
4. **Memory Usage**: Loading all data at once
5. **Cache Misses**: Redundant API calls

---

## 🔧 Optimization Strategy

### Phase 1: Immediate Wins (This Week)

#### 1.1 Implement Aggressive Caching
**Goal**: Reduce API calls by 80-90%

**Implementation:**
```python
# In data_engine.py
class PriceCache:
    def __init__(self, ttl_seconds=3600):  # 1 hour TTL
        self._cache = {}
        self._ttl = ttl_seconds
    
    def get(self, symbol, days):
        key = f"{symbol}_{days}"
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return data
        return None
    
    def set(self, symbol, days, data):
        key = f"{symbol}_{days}"
        self._cache[key] = (data, time.time())
```

**Expected Impact**: 
- First scan: ~60 seconds (cold cache)
- Subsequent scans: ~5-10 seconds (warm cache)
- **80-90% faster** for repeated scans

---

#### 1.2 Parallel Processing with Ray
**Goal**: Utilize all CPU cores for symbol processing

**Current**: Some parallelism with `concurrent.futures`
**Target**: Full Ray integration for distributed computing

**Implementation:**
```python
import ray

@ray.remote
def process_symbol(symbol, config):
    """Process single symbol in parallel"""
    try:
        # Fetch data
        history = get_price_history(symbol, config.lookback_days)
        
        # Compute indicators
        scores = compute_scores(history)
        
        # Compute factors
        factors = compute_factor_bundle(history)
        
        # Build result
        return build_result_row(symbol, scores, factors)
    except Exception as e:
        logger.warning(f"Failed to process {symbol}: {e}")
        return None

def run_scan_parallel(symbols, config):
    """Scan symbols in parallel with Ray"""
    ray.init(ignore_reinit_error=True)
    
    # Distribute work across cores
    futures = [process_symbol.remote(sym, config) for sym in symbols]
    
    # Gather results
    results = ray.get(futures)
    
    # Filter out failures
    return [r for r in results if r is not None]
```

**Expected Impact**:
- **4-8x faster** on multi-core systems
- Scales to 16+ cores
- Can distribute to cluster if needed

---

#### 1.3 Optimize Indicator Computation
**Goal**: Reduce computation time per symbol

**Strategies:**
1. **Vectorize Operations**: Use NumPy/Pandas vectorization
2. **Lazy Evaluation**: Only compute needed indicators
3. **Precompute Common Values**: Cache intermediate results
4. **Use Numba JIT**: Compile hot paths to machine code

**Implementation:**
```python
from numba import jit

@jit(nopython=True)
def fast_rsi(prices, period=14):
    """JIT-compiled RSI calculation"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi
```

**Expected Impact**:
- **2-3x faster** indicator calculations
- Lower CPU usage
- Better battery life on mobile

---

### Phase 2: Advanced Optimizations (Next Week)

#### 2.1 Implement Redis Cache
**Goal**: Persistent, fast cache across sessions

**Setup:**
```bash
# Install Redis
pip install redis

# Start Redis server
redis-server
```

**Implementation:**
```python
import redis
import pickle

class RedisCache:
    def __init__(self):
        self.client = redis.Redis(host='localhost', port=6379, db=0)
    
    def get_price_history(self, symbol, days):
        key = f"price:{symbol}:{days}"
        data = self.client.get(key)
        if data:
            return pickle.loads(data)
        return None
    
    def set_price_history(self, symbol, days, data, ttl=3600):
        key = f"price:{symbol}:{days}"
        self.client.setex(key, ttl, pickle.dumps(data))
```

**Expected Impact**:
- **Persistent cache** across app restarts
- **Sub-second** cache hits
- **Shared cache** across multiple users/instances

---

#### 2.2 Batch API Requests
**Goal**: Reduce API call overhead

**Current**: One API call per symbol
**Target**: Batch requests for multiple symbols

**Implementation:**
```python
def fetch_batch_prices(symbols, days=90):
    """Fetch prices for multiple symbols in one request"""
    # Polygon supports batch requests
    batch_size = 50
    results = {}
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        # Make single API call for batch
        response = polygon_client.get_grouped_daily(batch, days)
        results.update(response)
    
    return results
```

**Expected Impact**:
- **10-20x fewer** API calls
- **Faster** overall scan time
- **Lower** API costs

---

#### 2.3 Incremental Scanning
**Goal**: Only scan changed symbols

**Strategy:**
```python
def incremental_scan(config):
    """Only scan symbols that have new data"""
    # Load last scan timestamp
    last_scan = load_last_scan_time()
    
    # Get symbols with updates since last scan
    updated_symbols = get_updated_symbols(since=last_scan)
    
    # Scan only updated symbols
    new_results = scan_symbols(updated_symbols, config)
    
    # Merge with cached results
    cached_results = load_cached_results()
    final_results = merge_results(new_results, cached_results)
    
    return final_results
```

**Expected Impact**:
- **90%+ faster** for intraday rescans
- **Real-time** updates possible
- **Lower** resource usage

---

### Phase 3: Infrastructure Upgrades (Week 3-4)

#### 3.1 Move to AWS/GCP for Better Performance
**Goal**: Faster, more scalable infrastructure

**Current**: Render (shared resources, slower)
**Target**: AWS EC2 or GCP Compute Engine

**Benefits:**
- **Dedicated resources** (no sharing)
- **GPU support** for ML models
- **Auto-scaling** for load spikes
- **Lower latency** (better regions)
- **More memory** for caching

**Cost Comparison:**
- Render Pro Plus: $85/month (4GB RAM, shared CPU)
- AWS t3.xlarge: ~$120/month (16GB RAM, 4 vCPUs, dedicated)
- **Worth it** for performance

**See**: `RENDER_VS_AWS_COMPARISON.md` for details

---

#### 3.2 Implement Database for Persistent Storage
**Goal**: Fast access to historical data

**Options:**
1. **PostgreSQL**: Full-featured, ACID compliant
2. **TimescaleDB**: Optimized for time-series data
3. **ClickHouse**: Ultra-fast analytics

**Recommendation**: **TimescaleDB**
- Built on PostgreSQL
- Optimized for time-series (perfect for stock data)
- Fast aggregations
- Compression for storage efficiency

**Implementation:**
```python
# Store daily prices
CREATE TABLE prices (
    symbol TEXT,
    date DATE,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume BIGINT,
    PRIMARY KEY (symbol, date)
);

# Create hypertable for time-series optimization
SELECT create_hypertable('prices', 'date');

# Query is blazing fast
SELECT * FROM prices 
WHERE symbol = 'AAPL' 
  AND date >= NOW() - INTERVAL '90 days'
ORDER BY date;
```

**Expected Impact**:
- **Sub-second** data retrieval
- **No API calls** for historical data
- **Unlimited** lookback periods
- **Cost savings** on API usage

---

### Phase 4: ML Model Optimization (Week 4-5)

#### 4.1 ONNX Runtime with GPU
**Goal**: 10-100x faster model inference

**Current**: Python models (CPU)
**Target**: ONNX Runtime with GPU acceleration

**Implementation:**
```python
import onnxruntime as ort

# Convert model to ONNX
import onnxmltools
onnx_model = onnxmltools.convert_lightgbm(lgb_model)

# Create inference session with GPU
session = ort.InferenceSession(
    onnx_model.SerializeToString(),
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# Batch inference (1000 symbols at once)
predictions = session.run(None, {'input': features_batch})
```

**Expected Impact**:
- **10-100x faster** predictions
- **Batch processing** of 1000+ symbols
- **Lower latency** for real-time scoring

---

#### 4.2 Feature Store for Precomputed Factors
**Goal**: Avoid recomputing factors every scan

**Implementation:**
```python
class FeatureStore:
    """Precomputed features for fast access"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def get_features(self, symbol, as_of_date):
        """Get precomputed features"""
        query = """
            SELECT momentum, value, quality, growth, volatility
            FROM features
            WHERE symbol = %s AND date = %s
        """
        return self.db.execute(query, (symbol, as_of_date))
    
    def update_features(self, symbol, date, features):
        """Store computed features"""
        query = """
            INSERT INTO features (symbol, date, momentum, value, quality, growth, volatility)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, date) DO UPDATE SET ...
        """
        self.db.execute(query, (symbol, date, *features))
```

**Expected Impact**:
- **5-10x faster** scans (no recomputation)
- **Consistent** features across scans
- **Historical** feature tracking

---

## 📊 Performance Targets

### Current Performance (Estimated):
- **Cold scan** (no cache): 60-120 seconds for 5,000 symbols
- **Warm scan** (with cache): 30-60 seconds
- **API calls**: 5,000+ per scan
- **Memory**: 2-4GB peak

### Target Performance (After Optimization):
- **Cold scan**: 10-15 seconds
- **Warm scan**: 2-5 seconds
- **Incremental scan**: <1 second
- **API calls**: 50-100 per scan (98% reduction)
- **Memory**: 1-2GB peak

### Stretch Goals:
- **Real-time scanning**: <500ms for top 100 symbols
- **Batch inference**: 10,000 symbols in <10 seconds
- **Zero API calls**: All data from database

---

## 🛠️ Implementation Roadmap

### Week 1: Caching & Parallelization
**Focus**: Quick wins for immediate performance boost

**Tasks:**
- [ ] Implement in-memory price cache with TTL
- [ ] Add Redis cache layer
- [ ] Optimize Ray parallelization
- [ ] Add batch API requests
- [ ] Benchmark improvements

**Expected Speedup**: 5-10x faster

---

### Week 2: Database Integration
**Focus**: Persistent storage for historical data

**Tasks:**
- [ ] Set up TimescaleDB instance
- [ ] Create schema for prices, fundamentals, events
- [ ] Implement data ingestion pipeline
- [ ] Update scanner to use database
- [ ] Add database caching layer

**Expected Speedup**: 10-20x faster (with warm database)

---

### Week 3: ML Model Optimization
**Focus**: GPU acceleration and batch inference

**Tasks:**
- [ ] Convert models to ONNX format
- [ ] Set up ONNX Runtime with GPU
- [ ] Implement batch inference
- [ ] Create feature store
- [ ] Optimize feature computation

**Expected Speedup**: 10-100x faster for ML predictions

---

### Week 4: Infrastructure Upgrade
**Focus**: Move to AWS/GCP for better resources

**Tasks:**
- [ ] Provision AWS EC2 instance (GPU-enabled)
- [ ] Set up TimescaleDB on RDS
- [ ] Configure Redis on ElastiCache
- [ ] Deploy optimized scanner
- [ ] Set up monitoring (CloudWatch/Datadog)

**Expected Speedup**: 2-5x faster (better hardware)

---

## 📈 Combined Performance Gains

### Optimization Stack:
1. **Caching** (5-10x): In-memory + Redis
2. **Parallelization** (4-8x): Ray with all cores
3. **Database** (10-20x): TimescaleDB vs API calls
4. **ML Optimization** (10-100x): ONNX + GPU
5. **Infrastructure** (2-5x): AWS vs Render

### Total Potential Speedup:
- **Conservative**: 50-100x faster
- **Realistic**: 100-500x faster
- **Optimistic**: 1000x+ faster

### Real-World Impact:
- **Current**: 60 seconds for 5,000 symbols
- **After optimization**: 0.5-5 seconds for 5,000 symbols
- **Real-time**: Possible for top 500 symbols

---

## 🎯 Detailed Implementation Plan

### Task 1: Implement Multi-Layer Caching

**File**: `technic_v4/data_engine.py`

**Add:**
```python
class MultiLayerCache:
    """Three-tier caching: Memory → Redis → Database"""
    
    def __init__(self):
        self.memory_cache = {}  # L1: In-memory (fastest)
        self.redis_cache = RedisCache()  # L2: Redis (fast)
        self.db_cache = DatabaseCache()  # L3: Database (persistent)
    
    def get_price_history(self, symbol, days):
        # Try L1 (memory)
        if symbol in self.memory_cache:
            return self.memory_cache[symbol]
        
        # Try L2 (Redis)
        data = self.redis_cache.get(symbol, days)
        if data:
            self.memory_cache[symbol] = data  # Promote to L1
            return data
        
        # Try L3 (database)
        data = self.db_cache.get(symbol, days)
        if data:
            self.redis_cache.set(symbol, days, data)  # Promote to L2
            self.memory_cache[symbol] = data  # Promote to L1
            return data
        
        # Cache miss - fetch from API
        data = fetch_from_polygon(symbol, days)
        
        # Populate all cache layers
        self.db_cache.set(symbol, days, data)
        self.redis_cache.set(symbol, days, data)
        self.memory_cache[symbol] = data
        
        return data
```

**Testing:**
```python
# Benchmark caching
import time

# Cold cache
start = time.time()
result1 = cache.get_price_history('AAPL', 90)
cold_time = time.time() - start
print(f"Cold cache: {cold_time:.2f}s")

# Warm cache (memory)
start = time.time()
result2 = cache.get_price_history('AAPL', 90)
warm_time = time.time() - start
print(f"Warm cache: {warm_time:.4f}s")
print(f"Speedup: {cold_time / warm_time:.0f}x")
```

---

### Task 2: Optimize Ray Parallelization

**File**: `technic_v4/scanner_core.py`

**Current Issue**: Not fully utilizing Ray's capabilities

**Optimization:**
```python
import ray
from ray.util.multiprocessing import Pool

def _run_symbol_scans_optimized(config, universe, ...):
    """Optimized parallel scanning with Ray"""
    
    # Initialize Ray with all cores
    if not ray.is_initialized():
        ray.init(num_cpus=os.cpu_count())
    
    # Create remote function
    @ray.remote
    def scan_symbol_remote(row, config, regime_tags, lookback):
        return _scan_single_symbol(row, config, regime_tags, lookback)
    
    # Distribute work
    futures = []
    for _, row in universe.iterrows():
        future = scan_symbol_remote.remote(row, config, regime_tags, effective_lookback)
        futures.append(future)
    
    # Gather results with progress
    results = []
    for i, future in enumerate(futures):
        try:
            result = ray.get(future, timeout=30)
            if result is not None:
                results.append(result)
            
            # Progress callback
            if progress_cb and i % 100 == 0:
                progress_cb("Scanning", i, len(futures))
        except Exception as e:
            logger.warning(f"Symbol scan failed: {e}")
    
    return pd.DataFrame(results)
```

**Expected Impact**:
- **Full CPU utilization** (all cores)
- **Better error handling** (timeouts)
- **Progress tracking** (user feedback)

---

### Task 3: Implement Smart Universe Filtering

**Goal**: Scan fewer symbols without losing quality

**Strategy:**
```python
def smart_filter_universe(universe, config):
    """Filter universe intelligently before scanning"""
    
    # Start with full universe
    working = universe.copy()
    
    # Filter 1: Remove illiquid stocks (saves 50% of symbols)
    working = working[working['avg_volume'] > 100000]
    
    # Filter 2: Remove penny stocks (saves 20% more)
    working = working[working['last_price'] > 5.0]
    
    # Filter 3: Remove stocks with no recent data (saves 10% more)
    working = working[working['last_update'] > (datetime.now() - timedelta(days=7))]
    
    # Filter 4: Sector focus (if specified)
    if config.sectors:
        working = working[working['sector'].isin(config.sectors)]
    
    # Result: 70-80% fewer symbols to scan
    logger.info(f"Filtered universe: {len(universe)} → {len(working)} symbols")
    return working
```

**Expected Impact**:
- **70-80% fewer** symbols to process
- **3-5x faster** scans
- **Better quality** results (liquid stocks only)

---

### Task 4: Implement Streaming Results

**Goal**: Show results as they come in (progressive loading)

**Implementation:**
```python
async def stream_scan_results(config):
    """Stream results as they're computed"""
    
    # Start scanning
    async for result in scan_symbols_async(config):
        # Yield result immediately
        yield result
        
        # Update progress
        progress = calculate_progress()
        yield {'type': 'progress', 'value': progress}
```

**Flutter Integration:**
```dart
Stream<ScanResult> streamScanResults() async* {
  final response = await http.post(
    Uri.parse('$baseUrl/v1/scan/stream'),
    headers: headers,
    body: jsonEncode(params),
  );
  
  // Parse streaming response
  await for (final chunk in response.stream) {
    final result = ScanResult.fromJson(jsonDecode(chunk));
    yield result;
  }
}
```

**Expected Impact**:
- **Perceived performance** boost (results appear faster)
- **Better UX** (progressive loading)
- **Lower memory** (don't hold all results)

---

## 🔬 Benchmarking Plan

### Metrics to Track:
1. **Total scan time** (end-to-end)
2. **API calls** (count and time)
3. **Cache hit rate** (%)
4. **CPU usage** (%)
5. **Memory usage** (MB)
6. **Results quality** (precision, recall)

### Benchmark Script:
```python
# File: technic_v4/dev/benchmark_scanner.py

import time
import psutil
from technic_v4.scanner_core import run_scan

def benchmark_scan(config, iterations=3):
    """Benchmark scanner performance"""
    
    results = []
    for i in range(iterations):
        # Clear caches for cold run
        if i == 0:
            clear_all_caches()
        
        # Measure
        start = time.time()
        start_mem = psutil.Process().memory_info().rss / 1024 / 1024
        
        df, msg = run_scan(config)
        
        end_time = time.time() - start
        end_mem = psutil.Process().memory_info().rss / 1024 / 1024
        
        results.append({
            'iteration': i,
            'time': end_time,
            'memory_mb': end_mem - start_mem,
            'results_count': len(df),
            'cache_state': 'cold' if i == 0 else 'warm'
        })
    
    return pd.DataFrame(results)

# Run benchmark
config = ScanConfig(max_symbols=5000)
results = benchmark_scan(config)
print(results)
```

---

## 📋 Optimization Checklist

### Immediate (This Week):
- [ ] Add in-memory caching with TTL
- [ ] Optimize Ray parallelization
- [ ] Implement smart universe filtering
- [ ] Add batch API requests
- [ ] Benchmark improvements

### Short-term (Next 2 Weeks):
- [ ] Set up Redis cache
- [ ] Implement incremental scanning
- [ ] Add streaming results
- [ ] Optimize indicator computation
- [ ] Add Numba JIT compilation

### Medium-term (Week 3-4):
- [ ] Set up TimescaleDB
- [ ] Migrate to AWS/GCP
- [ ] Implement feature store
- [ ] Add ONNX Runtime with GPU
- [ ] Full end-to-end optimization

### Long-term (Month 2+):
- [ ] Real-time streaming data
- [ ] Distributed scanning (multi-node)
- [ ] Advanced ML models
- [ ] Custom indicator engine
- [ ] Portfolio optimization

---

## 🎯 Success Criteria

### Performance:
- ✅ Cold scan: <15 seconds (from 60s)
- ✅ Warm scan: <5 seconds (from 30s)
- ✅ Incremental: <1 second
- ✅ API calls: <100 per scan (from 5,000)
- ✅ Memory: <2GB (from 4GB)

### Quality:
- ✅ Same or better results
- ✅ No accuracy loss
- ✅ Stable and reliable
- ✅ Graceful error handling

### User Experience:
- ✅ Feels instant (<3 seconds)
- ✅ Progressive loading
- ✅ Real-time updates
- ✅ Smooth, responsive UI

---

## 💡 Quick Wins to Start Today

### 1. Add Simple In-Memory Cache (30 minutes)
```python
# In data_engine.py
_price_cache = {}
_cache_ttl = 3600  # 1 hour

def get_price_history_cached(symbol, days):
    key = f"{symbol}_{days}"
    now = time.time()
    
    if key in _price_cache:
        data, timestamp = _price_cache[key]
        if now - timestamp < _cache_ttl:
            return data
    
    # Fetch from API
    data = get_price_history(symbol, days)
    _price_cache[key] = (data, now)
    return data
```

**Impact**: 5-10x faster for repeated scans

---

### 2. Optimize Universe Filtering (15 minutes)
```python
# In scanner_core.py
def _filter_universe(universe, config):
    # Add liquidity filter
    universe = universe[universe['avg_volume'] > 100000]
    
    # Add price filter
    universe = universe[universe['last_price'] > 5.0]
    
    # Result: 70% fewer symbols
    return universe
```

**Impact**: 3x faster scans

---

### 3. Enable Ray Logging (5 minutes)
```python
# See what Ray is doing
import ray
ray.init(logging_level=logging.INFO)
```

**Impact**: Better visibility into parallelization

---

## 📊 Monitoring & Metrics

### Add Performance Logging:
```python
# In scanner_core.py
logger.info(f"[PERF] Universe loaded: {len(universe)} symbols")
logger.info(f"[PERF] After filtering: {len(working)} symbols")
logger.info(f"[PERF] Data fetching: {fetch_time:.2f}s")
logger.info(f"[PERF] Indicator computation: {indicator_time:.2f}s")
logger.info(f"[PERF] MERIT computation: {merit_time:.2f}s")
logger.info(f"[PERF] Total scan time: {total_time:.2f}s")
logger.info(f"[PERF] Cache hit rate: {cache_hits/total_requests*100:.1f}%")
```

### Add Metrics Dashboard:
- Track scan times over time
- Monitor cache hit rates
- Alert on performance degradation
- Compare before/after optimizations

---

## 🎉 Expected Outcomes

### After Week 1:
- **10x faster** scans with caching
- **Better parallelization** with Ray
- **Fewer API calls** with batching

### After Week 2:
- **20x faster** with database
- **Persistent cache** across restarts
- **Lower API costs**

### After Week 3:
- **100x faster** ML predictions
- **Feature store** for instant access
- **Real-time** scanning possible

### After Week 4:
- **Production-grade** infrastructure
- **Auto-scaling** for load
- **Sub-second** scans for top symbols

---

**Next Action**: Choose which optimization to start with. I recommend starting with Task 1 (Multi-Layer Caching) for immediate 5-10x performance boost.
