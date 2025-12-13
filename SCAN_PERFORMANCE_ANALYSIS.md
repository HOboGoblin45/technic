# 🐌 Scan Performance Analysis & Solutions

## Current Performance

Based on the Render logs, scanning 5,277 symbols takes **~54 minutes (3,242 seconds)**:

```
[SCAN PERF] symbol engine: 5277 symbols via threadpool in 3235.45s 
(0.613s/symbol, max_workers=10, use_ray=False)
```

**Performance**: 0.613 seconds per symbol × 5,277 symbols = 54 minutes

---

## 🎯 Why It's Slow

### 1. **Render Free Tier Limitations**
- **CPU**: Limited to 0.5 CPU cores (shared)
- **RAM**: 512 MB
- **Cold starts**: Service spins down after inactivity
- **Network**: Slower I/O for API calls (Polygon, etc.)

### 2. **Sequential Processing**
- Current: `max_workers=10` threadpool
- Render's limited CPU can't fully utilize 10 workers
- Each symbol requires:
  - Price history fetch (Polygon API)
  - Technical indicator calculations
  - ML model inference (XGBoost)
  - Factor computations

### 3. **No Ray Acceleration**
- Log shows: `use_ray=False`
- Ray would enable distributed processing
- But Render free tier doesn't support Ray clusters

---

## 🚀 Solutions (Ranked by Effectiveness)

### Option 1: Reduce Scan Scope (IMMEDIATE - No Cost)

**Reduce max_symbols from 6000 to 100-500:**

```dart
// In api_service.dart, change default:
'max_symbols': int.tryParse(params?['max_symbols'] ?? '500') ?? 500,
```

**Impact**:
- 500 symbols × 0.613s = **5 minutes** (vs 54 minutes)
- 100 symbols × 0.613s = **1 minute** (vs 54 minutes)

**Trade-off**: Fewer stocks scanned, but much faster results

---

### Option 2: Upgrade Render Plan (RECOMMENDED - $7-25/month)

**Render Starter Plan ($7/month)**:
- 1 CPU core (2x faster)
- 1 GB RAM (2x more)
- No cold starts
- **Expected**: 15-20 minutes for full scan

**Render Standard Plan ($25/month)**:
- 2 CPU cores (4x faster)
- 2 GB RAM (4x more)
- **Expected**: 8-10 minutes for full scan

---

### Option 3: Enable Ray + GPU (BEST - Requires Infrastructure)

**Deploy to AWS/GCP with Ray cluster:**
- Use Ray for distributed processing
- GPU acceleration for ML models
- **Expected**: 2-5 minutes for full scan

**Cost**: $50-200/month depending on instance types

**Setup Required**:
1. Deploy to AWS EC2 or GCP Compute
2. Set up Ray cluster (3-5 nodes)
3. Enable GPU instances for ML inference
4. Configure auto-scaling

---

### Option 4: Optimize Backend Code (MEDIUM EFFORT)

**Improvements to make**:

1. **Cache More Aggressively**:
   - Cache price data for 1 hour instead of per-request
   - Cache ML model predictions
   - **Gain**: 20-30% faster

2. **Batch API Calls**:
   - Fetch multiple symbols' data in one Polygon API call
   - **Gain**: 30-40% faster

3. **Reduce Lookback Period**:
   - Default 150 days → 90 days
   - Less data to process
   - **Gain**: 15-20% faster

4. **Lazy Load Features**:
   - Only compute features needed for scoring
   - Skip expensive calculations for low-ranked stocks
   - **Gain**: 25-35% faster

**Combined**: Could reduce to 20-25 minutes on Render free tier

---

## 💡 Recommended Approach

### For Development/Testing (NOW):
**Reduce max_symbols to 100-500**
- Instant fix, no cost
- Fast enough for testing
- Can increase later

### For Production (SOON):
**Upgrade to Render Starter ($7/month)**
- 2x faster (15-20 min scans)
- No cold starts
- Reliable performance
- Still affordable

### For Scale (FUTURE):
**AWS/GCP with Ray + GPU**
- Professional-grade performance
- 2-5 minute scans
- Supports thousands of users
- Required for App Store launch

---

## 🔧 Quick Fix (Reduce max_symbols)

I can immediately reduce max_symbols to make scans faster. Choose one:

**A. Conservative (100 symbols)**: ~1 minute scans
**B. Balanced (500 symbols)**: ~5 minute scans  
**C. Aggressive (1000 symbols)**: ~10 minute scans
**D. Keep 6000**: ~54 minute scans (current)

---

## 📊 Performance Comparison

| Solution | Scan Time | Cost | Effort |
|----------|-----------|------|--------|
| Reduce to 100 symbols | 1 min | $0 | 1 min |
| Reduce to 500 symbols | 5 min | $0 | 1 min |
| Render Starter | 15-20 min | $7/mo | 5 min |
| Render Standard | 8-10 min | $25/mo | 5 min |
| Code optimization | 20-25 min | $0 | 2-3 days |
| AWS + Ray + GPU | 2-5 min | $50-200/mo | 1-2 weeks |

---

## ✅ My Recommendation

**For right now**: Reduce max_symbols to 500
- Fast enough to test (5 minutes)
- Still scans a good sample
- No cost, instant fix

**For production**: Upgrade to Render Starter ($7/month)
- Professional performance
- Reliable and fast
- Worth it for a production app

**Would you like me to reduce max_symbols to 500 for faster testing?**
</result>
</attempt_completion>
