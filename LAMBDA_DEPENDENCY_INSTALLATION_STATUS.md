# Lambda Dependency Installation Status

## Current Action

Installing Lambda dependencies directly into `lambda_deploy` directory:

```powershell
pip install -r requirements_lambda.txt -t lambda_deploy
```

## Why This Is Needed

The `deploy_lambda.ps1` script created the ZIP file but skipped the dependency installation step. This is why the Lambda test failed with "No module named 'redis'".

## What's Being Installed

From `requirements_lambda.txt`:
- ✅ numpy==1.24.3
- ✅ pandas==2.0.3
- ✅ requests==2.31.0
- ✅ polygon-api-client==1.12.5
- ✅ redis==5.0.0 (This is the missing one!)
- ✅ boto3==1.34.0
- 🔄 ta-lib==0.4.28 (Currently installing - may take a few minutes)
- ⏳ scikit-learn==1.3.0
- ⏳ scipy==1.11.3

## Expected Timeline

- **ta-lib:** 2-3 minutes (requires compilation)
- **scikit-learn:** 1-2 minutes
- **scipy:** 1-2 minutes
- **Total:** ~5-7 minutes

## After Installation Completes

### Step 1: Verify Redis Is Installed

```powershell
dir lambda_deploy\redis
```

Should show the redis directory with Python files.

### Step 2: Recreate ZIP File

```powershell
# Remove old ZIP
Remove-Item technic-scanner.zip -Force

# Create new ZIP with dependencies
Compress-Archive -Path lambda_deploy\* -DestinationPath technic-scanner.zip
```

### Step 3: Upload to Lambda

```powershell
aws lambda update-function-code `
  --function-name technic-scanner `
  --zip-file fileb://technic-scanner.zip
```

Or use S3 upload script:

```powershell
.\upload_lambda_via_s3.ps1
```

### Step 4: Test Lambda

1. Go to AWS Lambda Console
2. Click "Test" button
3. Should now work without Redis import error
4. First run: 30-60s
5. Second run: 1-2s (Redis cache)

## Expected Package Size

With all dependencies:
- **Before:** 113.91 MB (without dependencies)
- **After:** ~150-200 MB (with all dependencies)

This is still under Lambda's 250 MB unzipped limit.

## Troubleshooting

### If ta-lib Installation Fails

ta-lib requires C++ build tools. If it fails:

**Option 1: Skip ta-lib (if not critical)**
```powershell
# Install without ta-lib
pip install numpy==1.24.3 pandas==2.0.3 requests==2.31.0 polygon-api-client==1.12.5 redis==5.0.0 boto3==1.34.0 scikit-learn==1.3.0 scipy==1.11.3 -t lambda_deploy
```

**Option 2: Use pre-built wheel**
Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

### If Package Is Too Large

If the final package exceeds 250 MB unzipped:

1. Remove unnecessary files:
```powershell
# Remove test files
Remove-Item lambda_deploy\*\tests -Recurse -Force
Remove-Item lambda_deploy\*\*.dist-info -Recurse -Force
```

2. Use Lambda layers for common libraries

## Next Steps After Successful Upload

1. **Test Lambda Function**
   - Verify Redis connection works
   - Check scan completes successfully
   - Confirm caching works (second run <2s)

2. **Integrate with Render**
   - Add AWS credentials to Render
   - Deploy api_hybrid.py
   - Test end-to-end

3. **Monitor Performance**
   - Check CloudWatch logs
   - Monitor cache hit rate
   - Verify costs

See `LAMBDA_TESTING_AND_RENDER_INTEGRATION.md` for detailed steps.

---

**Current Status:** Installing dependencies... (ta-lib in progress)
