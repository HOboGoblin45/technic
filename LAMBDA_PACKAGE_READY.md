# Lambda Package Creation - In Progress

## ✅ Status: Creating ZIP with All Dependencies

The deployment package is being created with:
- ✅ Redis library installed
- ✅ NumPy, Pandas, SciPy, Scikit-learn
- ✅ Boto3 (AWS SDK)
- ✅ Polygon API client
- ✅ All other required dependencies

## Current Action

```powershell
Compress-Archive -Path lambda_deploy\* -DestinationPath technic-scanner.zip -CompressionLevel Optimal
```

This is compressing ~150-200 MB of files, which takes 2-3 minutes.

## What's Included

### Core Lambda Function
- `lambda_function.py` - Main handler with Redis caching
- `technic_v4/` - Complete scanner module

### Dependencies (All Installed)
- ✅ **redis==5.0.0** - Redis client (THIS WAS MISSING BEFORE!)
- ✅ numpy==1.24.3 - Numerical computing
- ✅ pandas==2.0.3 - Data manipulation
- ✅ scipy==1.11.3 - Scientific computing
- ✅ scikit-learn==1.3.0 - Machine learning
- ✅ boto3==1.34.0 - AWS SDK
- ✅ polygon-api-client==1.12.5 - Market data
- ✅ requests==2.31.0 - HTTP client
- ✅ And all their dependencies

### What Was Skipped
- ❌ ta-lib - Requires C++ compilation (not critical for Lambda)

## Next Steps (Once ZIP Completes)

### 1. Check ZIP Size
```powershell
Get-Item technic-scanner.zip | Select-Object Name, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB,2)}}
```

Expected: 40-60 MB (compressed from ~150-200 MB uncompressed)

### 2. Upload to Lambda

**Option A: AWS Console (if ZIP < 50MB)**
1. Go to AWS Lambda Console
2. Open `technic-scanner` function
3. Click "Upload from" → ".zip file"
4. Select `technic-scanner.zip`
5. Click "Save"
6. Wait 2-3 minutes for upload

**Option B: AWS CLI (if ZIP > 50MB or Console fails)**
```powershell
aws lambda update-function-code `
  --function-name technic-scanner `
  --zip-file fileb://technic-scanner.zip
```

**Option C: S3 Upload (for large files)**
```powershell
.\upload_lambda_via_s3.ps1
```

### 3. Test Lambda Function

1. Go to Lambda Console → Test tab
2. Use existing test event or create new one:
```json
{
  "sectors": ["Technology"],
  "max_symbols": 5,
  "min_tech_rating": 10.0
}
```
3. Click "Test"
4. Wait 30-60 seconds
5. Check results

### 4. Verify Redis Connection

Check CloudWatch logs for:
```
[LAMBDA] Connected to Redis Cloud
```

If you see this, Redis is working! ✅

### 5. Test Cache

Run the same test twice:
- **First run:** 30-60 seconds (computation)
- **Second run:** 1-2 seconds (Redis cache hit)

You should see in logs:
```
[LAMBDA] Cache hit for key: lambda_scan:...
```

## Expected Results

### Performance
- **Cold start:** 30-60s (first invocation)
- **Warm start:** 20-40s (uncached scan)
- **Cache hit:** 1-2s (cached scan)

### Logs (CloudWatch)
```
[LAMBDA] Connected to Redis Cloud
[LAMBDA] Processing scan request...
[LAMBDA] Scan completed in 35.2s
[LAMBDA] Found 5 results
[LAMBDA] Cached result for 300s
```

### Response
```json
{
  "statusCode": 200,
  "body": {
    "cached": false,
    "source": "lambda",
    "results": {
      "symbols": [...],
      "status": "Scan complete",
      "metrics": {...},
      "scan_time": 35.2
    },
    "execution_time": 36.5,
    "lambda_info": {
      "memory_limit": 10240,
      "memory_used": 2500,
      "time_remaining": 863500
    }
  }
}
```

## Troubleshooting

### If Upload Fails
- **Error:** "Request entity too large"
- **Solution:** Use S3 upload method or AWS CLI

### If Test Fails with Redis Error
- **Error:** "Unable to import module 'lambda_function': No module named 'redis'"
- **Solution:** This should NOT happen now - Redis is installed!
- **Verify:** Check lambda_deploy/redis directory exists in ZIP

### If Test Times Out
- **Error:** "Task timed out after 15.00 seconds"
- **Solution:** Increase timeout to 15 minutes (900 seconds)
- **Check:** Configuration → General configuration → Timeout

### If Memory Error
- **Error:** "Runtime exited with error: signal: killed"
- **Solution:** Increase memory to 12GB or 14GB
- **Check:** Configuration → General configuration → Memory

## Integration with Render (After Lambda Works)

Once Lambda is tested and working:

1. **Add AWS Credentials to Render**
   ```
   AWS_ACCESS_KEY_ID = your_key
   AWS_SECRET_ACCESS_KEY = your_secret
   AWS_REGION = us-east-1
   USE_LAMBDA = true
   LAMBDA_FUNCTION_NAME = technic-scanner
   ```

2. **Deploy api_hybrid.py to Render**
   ```bash
   git add api_hybrid.py
   git commit -m "Add Lambda integration"
   git push origin main
   ```

3. **Test End-to-End**
   ```bash
   curl -X POST https://technic-m5vn.onrender.com/scan \
     -H "Content-Type: application/json" \
     -d '{"sectors": ["Technology"], "max_symbols": 5}'
   ```

## Success Metrics

✅ **Lambda test passes** (no Redis import error)
✅ **CloudWatch shows Redis connection**
✅ **First scan: 30-60s**
✅ **Second scan: 1-2s (cache hit)**
✅ **Cache hit rate: >70%**
✅ **No timeout errors**
✅ **No memory errors**

---

**Current Status:** Creating ZIP file... (almost done!)

**Next:** Upload to Lambda and test!
