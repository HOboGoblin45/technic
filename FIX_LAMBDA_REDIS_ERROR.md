# Fix Lambda Redis Import Error

## Problem Identified

Lambda test failed with error:
```
Unable to import module 'lambda_function': No module named 'redis'
```

**Root Cause:** The deployment package was created but the dependencies (including Redis) weren't properly installed into the `lambda_deploy` directory.

## Solution: Rebuild Lambda Package with Dependencies

### Step 1: Clean and Rebuild

Run the deployment script again to properly install all dependencies:

```powershell
.\deploy_lambda.ps1
```

This will:
1. Clean the lambda_deploy directory
2. Copy lambda_function.py and technic_v4 module
3. **Install all dependencies from requirements_lambda.txt** (including redis==5.0.0)
4. Create a new technic-scanner.zip

### Step 2: Verify Dependencies Are Included

After the script completes, check that Redis is installed:

```powershell
# Check if redis is in the package
dir lambda_deploy\redis*
```

You should see a `redis` directory with the Redis Python library.

### Step 3: Re-upload to Lambda

Since the package will be large (113+ MB), use AWS CLI:

```powershell
aws lambda update-function-code `
  --function-name technic-scanner `
  --zip-file fileb://technic-scanner.zip
```

Or use the S3 upload script:

```powershell
.\upload_lambda_via_s3.ps1
```

### Step 4: Test Again

1. Go to AWS Lambda Console
2. Click "Test" button
3. Should now work without the Redis import error

## Alternative: Make Redis Optional

If you want Lambda to work without Redis (no caching), modify the code to make Redis optional:

### Edit lambda_deploy/lambda_function.py

Change the import section:

```python
# At the top of the file
import json
import os
import time
from typing import Dict, Any, Optional

# Make redis import optional
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from technic_v4.scanner_core import run_scan, ScanConfig
from technic_v4.infra.logging import get_logger

logger = get_logger()

# Initialize Redis connection (shared with Render)
REDIS_URL = os.environ.get('REDIS_URL')
redis_client = None

if REDIS_AVAILABLE and REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        logger.info("[LAMBDA] Connected to Redis Cloud")
    except Exception as e:
        logger.error(f"[LAMBDA] Failed to connect to Redis: {e}")
        redis_client = None
elif not REDIS_AVAILABLE:
    logger.warning("[LAMBDA] Redis library not available - caching disabled")
else:
    logger.warning("[LAMBDA] No REDIS_URL provided - caching disabled")
```

This way, Lambda will work even without Redis, but caching will be disabled.

## Recommended Approach

**Use Step 1-4 above** to properly rebuild the package with all dependencies. This ensures:
- ✅ Redis caching works
- ✅ 2-second cached responses
- ✅ 70-85% cache hit rate
- ✅ Better performance overall

## Why This Happened

The `deploy_lambda.ps1` script may have encountered an error during dependency installation, or the dependencies weren't properly included in the ZIP file. Common causes:

1. **Network issues** during pip install
2. **Permission issues** writing to lambda_deploy directory
3. **Incomplete installation** of some packages
4. **Missing build tools** for some dependencies (like ta-lib)

## Verification Checklist

After rebuilding, verify:

- [ ] `lambda_deploy/redis/` directory exists
- [ ] `lambda_deploy/numpy/` directory exists
- [ ] `lambda_deploy/pandas/` directory exists
- [ ] `lambda_deploy/technic_v4/` directory exists
- [ ] `technic-scanner.zip` is created (should be 113+ MB)
- [ ] Upload to Lambda succeeds
- [ ] Test in Lambda Console passes
- [ ] CloudWatch logs show "Connected to Redis Cloud"

## Next Steps After Fix

Once Lambda is working with Redis:

1. **Test caching:**
   - First test: Should take 30-60s
   - Second test (same parameters): Should take 1-2s

2. **Integrate with Render:**
   - Add AWS credentials to Render
   - Deploy api_hybrid.py
   - Test end-to-end

3. **Monitor performance:**
   - Check cache hit rate
   - Verify response times
   - Monitor costs

See `LAMBDA_TESTING_AND_RENDER_INTEGRATION.md` for detailed testing steps.
