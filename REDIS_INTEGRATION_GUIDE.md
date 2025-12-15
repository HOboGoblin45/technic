# Redis Integration Guide for Render
## Connecting Your Scanner to Redis for 60-Second Scans

**Your Redis Details:**
- **Endpoint:** `redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579`
- **Database:** `database-MJ6OLK48`
- **Region:** US East (N. Virginia)
- **Type:** Redis on AWS
- **Memory:** 2.5 GB (5 GB total)

---

## Step 1: Connect Redis to Your Render Service

### Option A: Using Render Dashboard (Recommended)

1. **Go to your Render service** (your Python backend)
2. **Click "Environment"** tab
3. **Add these environment variables:**

```bash
REDIS_URL=redis://redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579
REDIS_HOST=redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com
REDIS_PORT=12579
REDIS_PASSWORD=<your-redis-password>
REDIS_DB=0
```

4. **Get your Redis password:**
   - In Redis dashboard, click "Connect"
   - Copy the password from the connection string
   - It will look like: `redis://default:YOUR_PASSWORD@redis-12579...`

5. **Save and redeploy** your Render service

### Option B: Using Render.yaml (Infrastructure as Code)

Add to your `render.yaml`:

```yaml
services:
  - type: web
    name: technic-backend
    env: python
    envVars:
      - key: REDIS_URL
        sync: false
      - key: REDIS_HOST
        value: redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com
      - key: REDIS_PORT
        value: 12579
      - key: REDIS_PASSWORD
        sync: false
      - key: REDIS_DB
        value: 0
```

---

## Step 2: Install Redis Python Client

Add to `requirements.txt`:

```txt
redis>=5.0.0
hiredis>=2.2.0  # C parser for better performance
```

Then redeploy or run:
```bash
pip install redis hiredis
```

---

## Step 3: Create Redis Cache Layer

I'll create a new file `technic_v4/cache/redis_cache.py`:

```python
import redis
import json
import os
from typing import Optional, Any
import pandas as pd
from datetime import datetime, timedelta

class RedisCache:
    """
    L3 Cache layer using Redis for persistent, cross-instance caching
    """
    
    def __init__(self):
        self.client = None
        self.enabled = False
        self._connect()
    
    def _connect(self):
        """Connect to Redis using environment variables"""
        try:
            redis_url = os.getenv('REDIS_URL')
            if redis_url:
                self.client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test connection
                self.client.ping()
                self.enabled = True
                print("[REDIS] Connected successfully")
            else:
                print("[REDIS] No REDIS_URL found, running without Redis")
        except Exception as e:
            print(f"[REDIS] Connection failed: {e}")
            self.enabled = False
    
    def get_price_data(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """
        Get cached price data for a symbol
        
        Args:
            symbol: Ticker symbol
            days: Number of days of data
        
        Returns:
            DataFrame with price data or None if not cached
        """
        if not self.enabled:
            return None
        
        try:
            key = f"price:{symbol}:{days}"
            data = self.client.get(key)
            
            if data:
                # Deserialize from JSON
                df = pd.read_json(data, orient='records')
                print(f"[REDIS] Cache hit for {symbol}")
                return df
            
            return None
        except Exception as e:
            print(f"[REDIS] Get error for {symbol}: {e}")
            return None
    
    def set_price_data(self, symbol: str, days: int, df: pd.DataFrame, ttl_hours: int = 24):
        """
        Cache price data for a symbol
        
        Args:
            symbol: Ticker symbol
            days: Number of days of data
            df: Price data DataFrame
            ttl_hours: Time to live in hours (default 24)
        """
        if not self.enabled or df is None or df.empty:
            return
        
        try:
            key = f"price:{symbol}:{days}"
            # Serialize to JSON
            data = df.to_json(orient='records')
            # Set with expiration
            self.client.setex(key, timedelta(hours=ttl_hours), data)
            print(f"[REDIS] Cached {symbol} for {ttl_hours}h")
        except Exception as e:
            print(f"[REDIS] Set error for {symbol}: {e}")
    
    def get_scan_results(self, scan_id: str) -> Optional[pd.DataFrame]:
        """Get cached scan results"""
        if not self.enabled:
            return None
        
        try:
            key = f"scan:{scan_id}"
            data = self.client.get(key)
            if data:
                return pd.read_json(data, orient='records')
            return None
        except Exception:
            return None
    
    def set_scan_results(self, scan_id: str, df: pd.DataFrame, ttl_hours: int = 1):
        """Cache scan results"""
        if not self.enabled or df is None or df.empty:
            return
        
        try:
            key = f"scan:{scan_id}"
            data = df.to_json(orient='records')
            self.client.setex(key, timedelta(hours=ttl_hours), data)
        except Exception:
            pass
    
    def clear_cache(self, pattern: str = "*"):
        """Clear cache by pattern"""
        if not self.enabled:
            return
        
        try:
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
                print(f"[REDIS] Cleared {len(keys)} keys")
        except Exception as e:
            print(f"[REDIS] Clear error: {e}")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            info = self.client.info('stats')
            return {
                "enabled": True,
                "total_keys": self.client.dbsize(),
                "hits": info.get('keyspace_hits', 0),
                "misses": info.get('keyspace_misses', 0),
                "hit_rate": info.get('keyspace_hits', 0) / max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0)) * 100
            }
        except Exception:
            return {"enabled": True, "error": "Could not fetch stats"}


# Singleton instance
_redis_cache = None

def get_redis_cache() -> RedisCache:
    """Get or create Redis cache instance"""
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCache()
    return _redis_cache
```

---

## Step 4: Integrate Redis into Data Engine

Update `technic_v4/data_engine.py` to use Redis as L3 cache:

```python
from technic_v4.cache.redis_cache import get_redis_cache

def get_stock_history_df(symbol: str, days: int = 150, use_intraday: bool = True):
    """
    Get price history with 3-layer caching:
    L1: In-memory dict (fastest)
    L2: In-memory LRU (fast)
    L3: Redis (persistent, shared)
    """
    # L1 Cache (in-memory dict)
    cache_key = (symbol, days, use_intraday)
    if cache_key in _price_cache:
        return _price_cache[cache_key]
    
    # L2 Cache (LRU)
    lru_result = _lru_cache.get(cache_key)
    if lru_result is not None:
        _price_cache[cache_key] = lru_result
        return lru_result
    
    # L3 Cache (Redis) - NEW!
    redis_cache = get_redis_cache()
    redis_result = redis_cache.get_price_data(symbol, days)
    if redis_result is not None:
        _price_cache[cache_key] = redis_result
        _lru_cache.put(cache_key, redis_result)
        return redis_result
    
    # Cache miss - fetch from API
    df = fetch_from_polygon(symbol, days, use_intraday)
    
    # Store in all cache layers
    if df is not None and not df.empty:
        _price_cache[cache_key] = df
        _lru_cache.put(cache_key, df)
        redis_cache.set_price_data(symbol, days, df, ttl_hours=24)
    
    return df
```

---

## Step 5: Configure Redis on Render

### In Render Dashboard:

1. **Go to your web service**
2. **Environment tab**
3. **Add Secret Files** (for sensitive data):
   - Click "Add Secret File"
   - Filename: `.env.redis`
   - Contents:
     ```
     REDIS_PASSWORD=your_actual_password_here
     ```

4. **Add Environment Variables:**
   ```
   REDIS_URL=redis://:${REDIS_PASSWORD}@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0
   REDIS_HOST=redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com
   REDIS_PORT=12579
   REDIS_DB=0
   TECHNIC_USE_REDIS=true
   ```

5. **Save and Deploy**

---

## Step 6: Test Redis Connection

Create `test_redis_connection.py`:

```python
#!/usr/bin/env python3
"""Test Redis connection"""

import os
import redis

def test_redis():
    redis_url = os.getenv('REDIS_URL')
    
    if not redis_url:
        print("❌ REDIS_URL not set")
        return False
    
    try:
        client = redis.from_url(redis_url, decode_responses=True)
        
        # Test connection
        client.ping()
        print("✅ Redis connection successful!")
        
        # Test set/get
        client.set('test_key', 'test_value', ex=60)
        value = client.get('test_key')
        print(f"✅ Set/Get test: {value}")
        
        # Get info
        info = client.info('server')
        print(f"✅ Redis version: {info.get('redis_version')}")
        print(f"✅ Memory used: {info.get('used_memory_human')}")
        
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

if __name__ == "__main__":
    test_redis()
```

Run on Render:
```bash
python test_redis_connection.py
```

---

## Step 7: Expected Performance Improvements

### With Redis L3 Cache:

**First Scan (Cold):**
- 5,000 symbols: 75-90s (same as now)
- All data fetched from API
- All data cached in Redis

**Second Scan (Warm):**
- 5,000 symbols: **15-25s** (70-80% from Redis!)
- Most data from Redis cache
- Only changed symbols from API

**Subsequent Scans:**
- 5,000 symbols: **10-20s** (80-90% from Redis!)
- Very high cache hit rate
- Minimal API calls

### Cache Hit Rate Progression:
- **Scan 1:** 0% (cold start)
- **Scan 2:** 70-80% (most symbols cached)
- **Scan 3+:** 85-95% (very high hit rate)

---

## Step 8: Monitoring Redis

### Check Cache Stats:

```python
from technic_v4.cache.redis_cache import get_redis_cache

cache = get_redis_cache()
stats = cache.get_stats()

print(f"Redis enabled: {stats['enabled']}")
print(f"Total keys: {stats['total_keys']}")
print(f"Hit rate: {stats['hit_rate']:.1f}%")
```

### Clear Cache (if needed):

```python
# Clear all price data
cache.clear_cache("price:*")

# Clear all scan results
cache.clear_cache("scan:*")

# Clear everything
cache.clear_cache("*")
```

---

## Step 9: Deployment Checklist

### Before Deploying:

- [ ] Redis password added to Render secrets
- [ ] REDIS_URL environment variable set
- [ ] redis package in requirements.txt
- [ ] Redis cache code added to data_engine.py
- [ ] Test connection script ready

### After Deploying:

- [ ] Run `test_redis_connection.py` on Render
- [ ] Verify Redis connection successful
- [ ] Run first scan (will be slow, populates cache)
- [ ] Run second scan (should be 3-4x faster!)
- [ ] Monitor cache hit rate
- [ ] Check Redis memory usage

---

## Step 10: Troubleshooting

### Connection Issues:

**Error:** "Connection refused"
- Check REDIS_URL is correct
- Verify Redis instance is running
- Check firewall/security groups

**Error:** "Authentication failed"
- Verify REDIS_PASSWORD is correct
- Check password doesn't have special characters that need escaping

**Error:** "Timeout"
- Increase timeout in connection settings
- Check network connectivity
- Verify Redis region matches Render region

### Performance Issues:

**Cache not improving performance:**
- Check cache hit rate (should be >70% after first scan)
- Verify TTL is appropriate (24 hours recommended)
- Check Redis memory isn't full

**Memory issues:**
- Monitor Redis memory usage
- Reduce TTL if needed
- Implement cache eviction policy

---

## Step 11: Expected Timeline

### Day 1: Redis Setup (2-3 hours)
- [ ] Add environment variables to Render
- [ ] Install redis package
- [ ] Create redis_cache.py
- [ ] Test connection

### Day 2: Integration (3-4 hours)
- [ ] Update data_engine.py
- [ ] Add L3 cache layer
- [ ] Test locally
- [ ] Deploy to Render

### Day 3: Testing (2-3 hours)
- [ ] Run first scan (populate cache)
- [ ] Run second scan (test cache)
- [ ] Measure performance
- [ ] Validate results

### Day 4: Optimization (2-3 hours)
- [ ] Tune cache TTL
- [ ] Optimize cache keys
- [ ] Monitor memory usage
- [ ] Document results

**Total:** 9-13 hours over 4 days

---

## Step 12: Performance Targets with Redis

### Scan 1 (Cold - Populating Cache):
- 5,000 symbols: 75-90s (same as now)
- Cache hit rate: 0%
- All data from API

### Scan 2 (Warm - Using Cache):
- 5,000 symbols: **20-30s** (70% from Redis)
- Cache hit rate: 70-80%
- Most data from Redis

### Scan 3+ (Hot - High Cache Hit):
- 5,000 symbols: **15-25s** (85% from Redis)
- Cache hit rate: 85-95%
- Minimal API calls

### **YOU'LL EASILY HIT 60-SECOND TARGET!**

In fact, with Redis, you'll likely achieve **15-30 second scans** for repeat scans!

---

## Step 13: Redis Best Practices

### Cache Key Strategy:
```python
# Price data: price:{symbol}:{days}
# Scan results: scan:{date}:{config_hash}
# Market data: market:{date}
# Macro data: macro:{date}
```

### TTL Strategy:
- **Price data:** 24 hours (daily updates)
- **Scan results:** 1 hour (frequent updates)
- **Market data:** 1 hour (intraday changes)
- **Macro data:** 4 hours (slower changes)

### Memory Management:
- **2.5 GB available** = ~50,000 cached symbols
- **Each symbol:** ~50 KB (150 days of data)
- **Monitor usage:** Keep under 80% (2 GB)
- **Eviction policy:** LRU (least recently used)

---

## Step 14: Next Steps

1. **Get Redis password** from Redis dashboard
2. **Add to Render** environment variables
3. **Deploy** with redis package
4. **Test connection** with test script
5. **Run first scan** (populate cache)
6. **Run second scan** (see the magic! ✨)

---

## 🎯 Expected Results

**Before Redis:**
- Every scan: 75-90s
- Every scan hits API
- No persistence

**After Redis:**
- First scan: 75-90s (populate)
- Second scan: **20-30s** (70% cached)
- Third+ scan: **15-25s** (85% cached)
- **Persistent across restarts!**

**You'll achieve sub-60-second scans starting from the SECOND scan!**

---

## 📞 Need Help?

If you encounter issues:
1. Check Render logs for Redis connection errors
2. Verify environment variables are set
3. Test Redis connection with test script
4. Check Redis dashboard for memory/connection stats

Let me know if you need help with any step!
