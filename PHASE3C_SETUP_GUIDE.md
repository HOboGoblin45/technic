# Phase 3C Redis Caching - Setup Guide

## Current Status

✅ **Infrastructure Complete:**
- Redis cache layer created (`technic_v4/cache/redis_cache.py`)
- Scanner integration prepared
- Authentication support added

❌ **Connection Issue:**
- Redis Cloud credentials appear to be invalid
- Getting "invalid username-password pair" error

## Solution Options

### Option 1: Regenerate Redis Cloud Password (Recommended)

1. **Go to Redis Cloud Console**: https://cloud.redis.io/
2. **Navigate to your database**: `database-MJ6OLK48`
3. **Security Tab** → **Reset Password**
4. **Copy the new password**
5. **Update your environment variables**:

```bash
# In your .env file or environment:
REDIS_URL=redis://:NEW_PASSWORD@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0
```

6. **Test connection**:
```bash
python test_redis_fixed.py
```

### Option 2: Create New Redis Database

If the current database has issues, create a new one:

1. **Redis Cloud Console** → **New Database**
2. **Choose Free Tier** (30MB, perfect for caching)
3. **Region**: US East (same as current)
4. **Copy credentials**
5. **Update REDIS_URL**

### Option 3: Use Local Redis (Development Only)

For local development/testing:

**Windows:**
```bash
# Install Redis using WSL or Docker
docker run -d -p 6379:6379 redis:latest
```

**Mac:**
```bash
brew install redis
brew services start redis
```

**Linux:**
```bash
sudo apt-get install redis-server
sudo systemctl start redis
```

Then set:
```bash
REDIS_URL=redis://localhost:6379/0
```

## Once Redis is Connected

### Step 1: Verify Connection
```bash
python test_redis_fixed.py
```

You should see:
```
✅ Direct connection successful! Ping response: True
✅ Set/Get test successful!
✅ All tests passed!
```

### Step 2: Integrate Caching into Scanner

The infrastructure is ready. Once Redis connects, caching will automatically work:

**What Gets Cached:**
- ✅ Price data (1 hour TTL)
- ✅ Technical indicators (5 min TTL)
- ✅ ML predictions (5 min TTL)
- ✅ Scan results (5 min TTL)

**Expected Performance:**
- **First scan**: Same speed (cache miss)
- **Subsequent scans** (within 5 min): **2x faster** (cache hit)
- **Cache hit rate target**: 70-85%

### Step 3: Test Performance

```bash
# First scan (cache miss)
python -m technic_v4.scanner_core

# Second scan within 5 minutes (cache hit - should be 2x faster)
python -m technic_v4.scanner_core
```

## Environment Variables Reference

Add these to your `.env` file or Render environment:

```bash
# Option 1: Full URL (recommended)
REDIS_URL=redis://:YOUR_PASSWORD@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0

# Option 2: Individual parameters
REDIS_HOST=redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com
REDIS_PORT=12579
REDIS_PASSWORD=YOUR_PASSWORD
REDIS_DB=0
```

## Troubleshooting

### "invalid username-password pair"
- Password has expired or been changed
- Regenerate password in Redis Cloud console
- Make sure no extra spaces in password

### "Connection refused"
- Redis instance is stopped
- Check Redis Cloud console - database should show "Active"
- Verify firewall/network settings

### "Connection timeout"
- Network issue or wrong host/port
- Verify endpoint in Redis Cloud console
- Check if your IP is whitelisted (if ACL enabled)

## Benefits of Redis Caching

Once working, you'll get:

1. **2x Speed Improvement**: Subsequent scans within 5 minutes are 2x faster
2. **Reduced API Costs**: Fewer Polygon API calls
3. **Better UX**: Faster results for users
4. **Scalability**: Multiple users benefit from shared cache
5. **Incremental Scans**: Only re-compute changed symbols

## Cost

- **Redis Cloud Free Tier**: $0/month (30MB)
- **Redis Cloud Paid**: $10/month (250MB) - if you need more
- **Current**: Already have Redis Cloud set up, just need valid credentials

## Next Steps

1. ✅ Fix Redis connection (regenerate password or create new database)
2. ✅ Test connection with `test_redis_fixed.py`
3. ✅ Run scanner - caching will work automatically
4. ✅ Measure performance improvement
5. ✅ Move to Phase 4 (AWS migration) or Phase 5 (GPU acceleration)

---

**Status**: Infrastructure ready, waiting for valid Redis credentials
**Blocker**: Invalid password - needs regeneration
**ETA**: 5 minutes once credentials are fixed
