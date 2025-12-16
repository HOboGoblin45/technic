# Redis Verification Guide - How to Know It's Working

## 🔍 How to Verify Redis is Working on Render

### **Method 1: Check Render Logs (EASIEST)**

**After deploying, check your Render logs for these messages:**

#### **✅ Good Signs (Redis Working):**

```
[CACHE] Redis connection successful
[CACHE] Connected to Redis at redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579
[CACHE] Redis client initialized
[CACHE] Cache hit for AAPL price data
[CACHE] Serving from cache: 1234 items
```

#### **❌ Bad Signs (Redis Not Working):**

```
[CACHE] Redis connection failed: Connection refused
[CACHE] Falling back to in-memory cache
[CACHE] Redis unavailable, using local cache
ERROR: Could not connect to Redis
```

---

### **Method 2: Test Redis Connection Directly**

**From Render Shell:**

1. Go to Render Dashboard → Your Service
2. Click **"Shell"** tab
3. Run this command:

```bash
redis-cli -u $REDIS_URL ping
```

**Expected output:**
```
PONG
```

**If you see `PONG`, Redis is working! ✅**

---

### **Method 3: Check Cache Performance (BEST TEST)**

**Run this test to see Redis in action:**

#### **Step 1: First Scan (Cold - No Cache)**

1. Open Flutter app
2. Run a scan
3. Note the time (should be ~75-90 seconds)
4. Check Render logs for:
   ```
   [CACHE] Cache miss for AAPL
   [CACHE] Fetching from Polygon API
   [CACHE] Cached AAPL for 3600s
   ```

#### **Step 2: Second Scan (Warm - With Cache)**

1. Run the SAME scan again immediately
2. Note the time (should be ~15-20 seconds)
3. Check Render logs for:
   ```
   [CACHE] Cache hit for AAPL
   [CACHE] Serving from cache
   [CACHE] Skipped API call (cached)
   ```

**If second scan is 4-5x faster, Redis is working! ✅**

---

### **Method 4: Check Redis Stats**

**From Render Shell, run:**

```bash
redis-cli -u $REDIS_URL INFO stats
```

**Look for:**
```
# Stats
total_connections_received:150
total_commands_processed:5000
instantaneous_ops_per_sec:25
keyspace_hits:4500
keyspace_misses:500
```

**Key metrics:**
- `keyspace_hits` > 0 = Cache is being used ✅
- `keyspace_hits` / (`keyspace_hits` + `keyspace_misses`) = Hit rate
- **Good hit rate:** > 80%

---

### **Method 5: Check Cached Keys**

**See what's actually cached:**

```bash
redis-cli -u $REDIS_URL KEYS "*"
```

**Expected output (example):**
```
1) "price:AAPL:90d"
2) "price:MSFT:90d"
3) "price:GOOGL:90d"
4) "ml_alpha:AAPL"
5) "ml_alpha:MSFT"
...
```

**If you see keys, Redis is storing data! ✅**

---

### **Method 6: Monitor Cache Size**

**Check how much data is cached:**

```bash
redis-cli -u $REDIS_URL DBSIZE
```

**Expected output:**
```
(integer) 5000
```

**This shows 5,000 cached items.**

**After first scan, this should be > 0 ✅**

---

## 🧪 Complete Verification Test

**Run this complete test to verify everything:**

### **Test Script:**

```bash
# 1. Test connection
echo "Testing Redis connection..."
redis-cli -u $REDIS_URL ping

# 2. Check stats
echo "Checking Redis stats..."
redis-cli -u $REDIS_URL INFO stats | grep keyspace

# 3. Check database size
echo "Checking cache size..."
redis-cli -u $REDIS_URL DBSIZE

# 4. List some keys
echo "Listing cached keys..."
redis-cli -u $REDIS_URL KEYS "*" | head -10
```

**Expected output:**
```
Testing Redis connection...
PONG

Checking Redis stats...
keyspace_hits:4500
keyspace_misses:500

Checking cache size...
(integer) 5000

Listing cached keys...
price:AAPL:90d
price:MSFT:90d
...
```

---

## 📊 Performance Indicators

### **Without Redis (Broken):**
- Every scan takes same time (~75-90 seconds)
- Logs show: "Fetching from API" every time
- No cache hits in logs

### **With Redis (Working):**
- First scan: 75-90 seconds
- Second scan: 15-20 seconds (4-5x faster!)
- Logs show: "Cache hit" messages
- `keyspace_hits` increasing in Redis stats

---

## 🚨 Troubleshooting

### **Issue: "Connection refused"**

**Cause:** Redis addon not active or wrong credentials

**Fix:**
1. Check Redis addon is active in Render
2. Verify `REDIS_URL` environment variable
3. Restart service

---

### **Issue: "No cache hits"**

**Cause:** Cache TTL too short or keys not matching

**Fix:**
1. Check `CACHE_TTL` environment variable (should be 3600)
2. Verify cache key format in logs
3. Check if cache is being cleared between scans

---

### **Issue: "Redis working but scans still slow"**

**Cause:** First scan always slow (cold cache)

**Expected behavior:**
- First scan: Full time (75-90 sec) - This is NORMAL
- Second scan: Fast (15-20 sec) - This proves Redis works

**Not a problem if:**
- Second scan is much faster
- Logs show cache hits

---

## 📝 Quick Checklist

**After deploying, verify these:**

- [ ] Render logs show "Redis connection successful"
- [ ] `redis-cli ping` returns `PONG`
- [ ] First scan completes (~75-90 seconds)
- [ ] Second scan is 4-5x faster (~15-20 seconds)
- [ ] Logs show "Cache hit" messages on second scan
- [ ] `DBSIZE` shows > 0 cached items
- [ ] `keyspace_hits` increasing in stats

**If all checked, Redis is working perfectly! ✅**

---

## 🎯 Expected Timeline

**After Manual Deploy:**

**Minute 0-2:** Service restarting
**Minute 2-3:** Redis connecting
**Minute 3:** Ready to test

**First Scan:**
- Time: 75-90 seconds
- Redis: Populating cache
- Logs: "Cache miss" → "Caching data"

**Second Scan (immediately after):**
- Time: 15-20 seconds
- Redis: Serving from cache
- Logs: "Cache hit" → "Serving from cache"

**This proves Redis is working! ✅**

---

## 💡 Pro Tips

### **Tip 1: Clear Cache to Test**

**Force a cold start:**
```bash
redis-cli -u $REDIS_URL FLUSHDB
```

**Then run scan - should be slow (cache empty)**

---

### **Tip 2: Monitor Cache Hit Rate**

**Check hit rate:**
```bash
redis-cli -u $REDIS_URL INFO stats | grep keyspace_hits
redis-cli -u $REDIS_URL INFO stats | grep keyspace_misses
```

**Calculate:**
```
Hit Rate = hits / (hits + misses) × 100%
```

**Good:** > 80%
**Excellent:** > 90%

---

### **Tip 3: Check Cache Expiry**

**See TTL of a key:**
```bash
redis-cli -u $REDIS_URL TTL "price:AAPL:90d"
```

**Output:**
```
(integer) 3456  # Seconds until expiry
```

**Should be ~3600 (1 hour) for fresh cache**

---

## 📞 Summary

**To verify Redis is working:**

1. **Check logs** for "Redis connection successful"
2. **Run `redis-cli ping`** - should return `PONG`
3. **Run 2 scans** - second should be 4-5x faster
4. **Check `DBSIZE`** - should be > 0 after first scan

**If all pass, Redis is working perfectly! ✅**

---

**After your manual deploy, check these and let me know what you see!**
