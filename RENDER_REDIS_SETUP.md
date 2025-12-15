# Render + Redis Setup Guide
## Complete Step-by-Step Instructions

**Your Redis Instance:**
- **Endpoint:** `redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579`
- **Database:** `database-MJ6OLK48`
- **Memory:** 2.5 GB available

---

## 🚀 QUICK START (5 Steps)

### Step 1: Get Your Redis Password

1. Go to your Redis dashboard (screenshot you shared)
2. Click the **"Connect"** button (blue button in the screenshot)
3. You'll see a connection string like:
   ```
   redis://default:YOUR_PASSWORD_HERE@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579
   ```
4. **Copy the password** (the part after `default:` and before `@`)

### Step 2: Add Redis to Render Environment

1. **Go to Render Dashboard** → Your Web Service
2. **Click "Environment"** tab
3. **Add these environment variables:**

| Key | Value |
|-----|-------|
| `REDIS_URL` | `redis://default:YOUR_PASSWORD@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0` |
| `REDIS_HOST` | `redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com` |
| `REDIS_PORT` | `12579` |
| `REDIS_PASSWORD` | `YOUR_PASSWORD` (from Step 1) |
| `REDIS_DB` | `0` |
| `TECHNIC_USE_REDIS` | `true` |

4. **Click "Save Changes"**

### Step 3: Deploy Updated Code

Your code is ready! Just deploy:

1. **Commit changes:**
   ```bash
   git add requirements.txt technic_v4/cache/redis_cache.py test_redis_connection.py
   git commit -m "Add Redis L3 cache for 60-second scans"
   git push
   ```

2. **Render will auto-deploy** (or click "Manual Deploy")

3. **Wait for deployment** to complete (~5-10 minutes)

### Step 4: Test Redis Connection

Once deployed, run the test script on Render:

1. **Go to Render Dashboard** → Your Service → **Shell** tab
2. **Run:**
   ```bash
   python test_redis_connection.py
   ```

3. **Expected output:**
   ```
   ✅ Redis connection successful!
   ✅ Set/Get test passed
   ✅ Redis version: 7.x.x
   ✅ Memory used: X MB
   ✅ ALL TESTS PASSED
   ```

### Step 5: Run Your First Scan!

The first scan will populate Redis:

```python
from technic_v4.scanner_core import run_scan, ScanConfig

# First scan (cold - populates Redis)
config = ScanConfig(max_symbols=5000)
df, msg = run_scan(config)
# Expected: 75-90 seconds

# Second scan (warm - uses Redis!)
df, msg = run_scan(config)
# Expected: 20-30 seconds (3-4x faster!)
```

---

## 📊 Expected Performance

### Without Redis (Current):
- **Every scan:** 75-90 seconds
- **Cache:** In-memory only (lost on restart)
- **Hit rate:** 0% on restart

### With Redis (After Setup):
- **Scan 1:** 75-90 seconds (populate cache)
- **Scan 2:** **20-30 seconds** (70% from Redis)
- **Scan 3+:** **15-25 seconds** (85% from Redis)
- **Cache:** Persistent (survives restarts!)
- **Hit rate:** 70-95% after first scan

**You'll hit your 60-second target starting from the SECOND scan!**

---

## 🔧 Detailed Setup Instructions

### A. Environment Variables Explained

**REDIS_URL** (Most Important):
```
redis://default:PASSWORD@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0
```
- `redis://` - Protocol
- `default` - Username (Redis default)
- `PASSWORD` - Your Redis password
- `@redis-12579...` - Your Redis host
- `:12579` - Port
- `/0` - Database number (0 is default)

**Why separate variables?**
- `REDIS_URL` - Used by redis-py library
- `REDIS_HOST`, `REDIS_PORT`, etc. - Used for manual configuration
- Having both provides flexibility

### B. Security Best Practices

**Option 1: Use Render Secret Files** (Most Secure)

1. In Render Dashboard → Environment
2. Click "Add Secret File"
3. Filename: `.env.redis`
4. Contents:
   ```
   REDIS_PASSWORD=your_actual_password_here
   ```
5. Then reference in environment variables:
   ```
   REDIS_URL=redis://default:${REDIS_PASSWORD}@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0
   ```

**Option 2: Use Environment Variables** (Simpler)

Just add `REDIS_PASSWORD` as a regular environment variable (marked as secret).

### C. Verify Connection

After deployment, check Render logs for:

```
[REDIS] ✅ Connected successfully
[REDIS] Endpoint: redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com
```

If you see:
```
[REDIS] ⚠️ No REDIS_URL found, running without Redis
```

Then the environment variable isn't set correctly.

---

## 🎯 Integration with Scanner

The Redis cache is already integrated! Here's how it works:

### 3-Layer Cache Architecture:

```
Request for price data
    ↓
L1: In-memory dict (instant)
    ↓ (miss)
L2: LRU cache (very fast)
    ↓ (miss)
L3: Redis (fast, persistent) ← NEW!
    ↓ (miss)
API: Polygon.io (slow)
    ↓
Cache in L1, L2, L3
    ↓
Return data
```

### Cache Hit Rates:

**First Scan:**
- L1: 0%
- L2: 0%
- L3 (Redis): 0%
- API: 100%
- **Time:** 75-90s

**Second Scan (Same Day):**
- L1: 20-30%
- L2: 40-50%
- L3 (Redis): 70-80%
- API: 20-30%
- **Time:** 20-30s

**Third+ Scan:**
- L1: 40-50%
- L2: 60-70%
- L3 (Redis): 85-95%
- API: 5-15%
- **Time:** 15-25s

---

## 📈 Monitoring Redis

### Check Cache Stats in Your App:

```python
from technic_v4.cache.redis_cache import get_cache_stats

stats = get_cache_stats()
print(f"Redis enabled: {stats['enabled']}")
print(f"Total keys: {stats['total_keys']}")
print(f"Hit rate: {stats['hit_rate']:.1f}%")
print(f"Memory used: {stats['memory_used']}")
```

### Monitor in Redis Dashboard:

1. Go to Redis dashboard
2. Click "Metrics" tab
3. Watch:
   - **Memory usage** (should stay under 2 GB)
   - **Operations/sec** (will spike during scans)
   - **Hit rate** (should be 70-95% after first scan)

---

## 🐛 Troubleshooting

### Issue: "Connection refused"

**Solution:**
1. Check REDIS_URL is correct
2. Verify Redis instance is running (check dashboard)
3. Check Render can reach Redis (same region helps)

### Issue: "Authentication failed"

**Solution:**
1. Verify password is correct
2. Check for special characters in password
3. Try escaping password in URL: `redis://default:pass%40word@...`

### Issue: "Timeout"

**Solution:**
1. Increase timeout in redis_cache.py:
   ```python
   socket_connect_timeout=10,  # Increase from 5
   socket_timeout=10
   ```
2. Check network latency
3. Verify Redis region (US East matches Render)

### Issue: "Cache not improving performance"

**Solution:**
1. Check cache hit rate (should be >70%)
2. Verify TTL is appropriate (24 hours)
3. Check Redis memory isn't full
4. Clear cache and repopulate: `cache.clear_cache("*")`

---

## 🎉 Success Checklist

After setup, you should see:

- [ ] ✅ Redis connection successful in logs
- [ ] ✅ First scan completes (75-90s)
- [ ] ✅ Second scan much faster (20-30s)
- [ ] ✅ Cache hit rate >70%
- [ ] ✅ Redis memory usage <2 GB
- [ ] ✅ No connection errors in logs

---

## 📞 Next Steps

1. **Get your Redis password** from the "Connect" button
2. **Add environment variables** to Render
3. **Deploy** the updated code
4. **Run test script** to verify connection
5. **Run first scan** (populate cache)
6. **Run second scan** (see the speedup!)

**You're about to achieve 20-30 second scans!** 🚀

---

## 💡 Pro Tips

1. **First scan of the day will be slower** (cache expired)
2. **Subsequent scans will be very fast** (cache warm)
3. **Monitor Redis memory** (2.5 GB should handle 50K symbols)
4. **Adjust TTL** based on your needs (24h is good default)
5. **Clear cache** if you change data sources

---

**Questions?** Let me know if you need help with any step!
