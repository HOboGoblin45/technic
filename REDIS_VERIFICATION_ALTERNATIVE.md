# Redis Verification - Alternative Methods (No redis-cli)

## ⚠️ Issue: redis-cli Not Available

Your Render environment doesn't have `redis-cli` installed. That's okay! Here are alternative ways to verify Redis is working.

---

## ✅ Method 1: Check Application Logs (BEST METHOD)

**This is the easiest and most reliable way!**

### **After deploying, check Render logs for:**

#### **✅ Redis Working:**
```
[CACHE] Redis connection successful
[CACHE] Connected to Redis
[CACHE] Cache initialized
```

#### **❌ Redis Not Working:**
```
[CACHE] Redis connection failed
[CACHE] Error connecting to Redis
ERROR: Connection refused
```

---

## ✅ Method 2: Performance Test (MOST RELIABLE)

**This proves Redis is working without any commands!**

### **Test Steps:**

1. **First Scan (Cold Cache):**
   - Open Flutter app
   - Run a scan
   - Time it: Should be ~75-90 seconds
   - Check logs: Should see "Cache miss" or "Fetching from API"

2. **Second Scan (Warm Cache):**
   - Run the SAME scan immediately
   - Time it: Should be ~15-20 seconds (4-5x faster!)
   - Check logs: Should see "Cache hit" or "Serving from cache"

**If second scan is much faster, Redis is working! ✅**

---

## ✅ Method 3: Install Redis Tools (Optional)

**If you really want redis-cli, add it to your Dockerfile:**

### **Update Dockerfile:**

```dockerfile
FROM python:3.11-slim

# Install redis-cli
RUN apt-get update && apt-get install -y redis-tools && rm -rf /var/lib/apt/lists/*

# Rest of your Dockerfile...
```

**Then redeploy.**

---

## ✅ Method 4: Python Script to Test Redis

**Create a test script to verify Redis:**

### **Create `test_redis.py`:**

```python
import os
import redis

def test_redis():
    redis_url = os.getenv('REDIS_URL')
    
    if not redis_url:
        print("❌ REDIS_URL not set")
        return False
    
    try:
        # Connect to Redis
        r = redis.from_url(redis_url)
        
        # Test connection
        r.ping()
        print("✅ Redis PING successful")
        
        # Test set/get
        r.set('test_key', 'test_value')
        value = r.get('test_key')
        print(f"✅ Redis SET/GET successful: {value}")
        
        # Get info
        info = r.info('stats')
        print(f"✅ Redis Stats:")
        print(f"   - Total connections: {info.get('total_connections_received', 0)}")
        print(f"   - Total commands: {info.get('total_commands_processed', 0)}")
        print(f"   - Keyspace hits: {info.get('keyspace_hits', 0)}")
        print(f"   - Keyspace misses: {info.get('keyspace_misses', 0)}")
        
        # Get database size
        dbsize = r.dbsize()
        print(f"✅ Redis DBSIZE: {dbsize} keys")
        
        # List some keys
        keys = r.keys('*')[:10]
        print(f"✅ Sample keys: {keys}")
        
        return True
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

if __name__ == '__main__':
    test_redis()
```

### **Run from Render Shell:**

```bash
python test_redis.py
```

**Expected output:**
```
✅ Redis PING successful
✅ Redis SET/GET successful: b'test_value'
✅ Redis Stats:
   - Total connections: 150
   - Total commands: 5000
   - Keyspace hits: 4500
   - Keyspace misses: 500
✅ Redis DBSIZE: 5000 keys
✅ Sample keys: [b'price:AAPL:90d', b'price:MSFT:90d', ...]
```

---

## ✅ Method 5: Check Logs for Cache Activity

**Look for these patterns in Render logs:**

### **First Scan (Populating Cache):**
```
[SCAN] Starting scan for 5000 symbols
[CACHE] Cache miss for AAPL
[CACHE] Fetching AAPL from Polygon API
[CACHE] Cached AAPL price data (TTL: 3600s)
[CACHE] Cache miss for MSFT
[CACHE] Fetching MSFT from Polygon API
[CACHE] Cached MSFT price data (TTL: 3600s)
...
[SCAN] Scan complete in 85.3 seconds
```

### **Second Scan (Using Cache):**
```
[SCAN] Starting scan for 5000 symbols
[CACHE] Cache hit for AAPL
[CACHE] Serving AAPL from cache
[CACHE] Cache hit for MSFT
[CACHE] Serving MSFT from cache
...
[SCAN] Scan complete in 18.7 seconds
```

**If you see "Cache hit" messages, Redis is working! ✅**

---

## 📊 Performance Indicators

### **Redis Working:**
- ✅ First scan: 75-90 seconds
- ✅ Second scan: 15-20 seconds (4-5x faster!)
- ✅ Logs show "Cache hit" messages
- ✅ Subsequent scans remain fast

### **Redis NOT Working:**
- ❌ All scans take same time (~75-90 seconds)
- ❌ Logs show "Cache miss" every time
- ❌ No "Cache hit" messages
- ❌ Logs show "Redis connection failed"

---

## 🎯 Recommended Verification Steps

**Do these in order:**

### **Step 1: Check Startup Logs**
1. Go to Render Dashboard → Logs
2. Look for: `[CACHE] Redis connection successful`
3. If you see it: ✅ Redis connected

### **Step 2: Run Performance Test**
1. Run scan from Flutter app
2. Note the time
3. Run scan again immediately
4. Compare times
5. If second is 4-5x faster: ✅ Redis working

### **Step 3: Check Cache Logs**
1. Look for "Cache hit" in logs during second scan
2. If you see them: ✅ Redis serving cached data

**If all 3 pass, Redis is working perfectly! ✅**

---

## 🚨 Troubleshooting

### **Issue: No "Cache" messages in logs**

**Possible causes:**
1. Logging level too high
2. Cache module not imported
3. Redis connection silently failing

**Fix:**
Add this to your environment variables:
```
LOG_LEVEL=DEBUG
```

---

### **Issue: "Cache miss" every time**

**Possible causes:**
1. Cache TTL too short
2. Keys not matching between scans
3. Cache being cleared

**Fix:**
Check environment variables:
```
CACHE_TTL=3600  # 1 hour
TECHNIC_USE_REDIS=1
```

---

### **Issue: Second scan not faster**

**Possible causes:**
1. Different scan parameters (different cache keys)
2. Cache expired between scans
3. Redis not actually being used

**Fix:**
1. Use exact same scan parameters
2. Run scans within 1 hour
3. Check logs for "Cache hit" messages

---

## 💡 Quick Verification Checklist

**After deploying, verify:**

- [ ] Render logs show "Redis connection successful"
- [ ] First scan completes (~75-90 seconds)
- [ ] Second scan is much faster (~15-20 seconds)
- [ ] Logs show "Cache hit" messages on second scan
- [ ] Subsequent scans remain fast

**If all checked, Redis is working! ✅**

---

## 📝 Summary

**You DON'T need redis-cli to verify Redis!**

**Best verification method:**
1. Run 2 scans back-to-back
2. Second should be 4-5x faster
3. Check logs for "Cache hit" messages

**That's it! Simple and reliable. ✅**

---

**Try the performance test now - run 2 scans and compare times!**
