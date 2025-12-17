# AWS Lambda Deployment - Next Steps

## ✅ Step 1: Create Deployment Package (In Progress)

**Status:** Running `deploy_lambda.ps1`

**What it's doing:**
1. ✅ Cleaning up old deployment
2. ✅ Creating deployment directory
3. ✅ Copying Lambda function (lambda_scanner.py → lambda_function.py)
4. ✅ Copying technic_v4 module
5. 🔄 Installing dependencies (in progress)
6. ⏳ Creating ZIP file

**Expected completion:** 3-5 minutes

---

## 📦 Step 2: Upload to AWS Lambda (Next)

Once the ZIP file is created, you'll need to:

### Option A: AWS Console (Recommended if ZIP < 50MB)

1. **Go to AWS Lambda Console**
   - URL: https://console.aws.amazon.com/lambda
   - Sign in with your AWS account

2. **Create Function**
   - Click "Create function"
   - Choose "Author from scratch"
   - Function name: `technic-scanner`
   - Runtime: `Python 3.11`
   - Architecture: `x86_64`
   - Click "Create function"

3. **Upload ZIP**
   - Click "Code" tab
   - Click "Upload from" → ".zip file"
   - Select `technic-scanner.zip` from your project directory
   - Click "Save"
   - Wait for upload to complete (1-2 minutes)

### Option B: AWS CLI (If ZIP > 50MB)

```powershell
# Upload to S3 first (you'll need to create a bucket)
aws s3 mb s3://technic-lambda-deploy
aws s3 cp technic-scanner.zip s3://technic-lambda-deploy/

# Update Lambda from S3
aws lambda update-function-code `
  --function-name technic-scanner `
  --s3-bucket technic-lambda-deploy `
  --s3-key technic-scanner.zip
```

---

## ⚙️ Step 3: Configure Lambda

### 3.1 Set Memory and Timeout

1. Go to Configuration → General configuration → Edit
2. Set these values:
   - **Memory:** `10240 MB` (10GB)
   - **Timeout:** `15 min 0 sec` (900 seconds)
3. Click "Save"

### 3.2 Add Environment Variables

1. Go to Configuration → Environment variables → Edit
2. Click "Add environment variable"
3. Add these variables:

```
Key: REDIS_URL
Value: redis://:ytvZ1OVXoGV4OenJH3GJYkDLeg2emqad@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0

Key: POLYGON_API_KEY
Value: [Your Polygon API key]
```

4. Click "Save"

---

## 🧪 Step 4: Test Lambda Function

### 4.1 Create Test Event

1. Go to "Test" tab
2. Click "Create new event"
3. Event name: `test-tech-scan`
4. Use this JSON:

```json
{
  "sectors": ["Technology"],
  "max_symbols": 5,
  "min_tech_rating": 10.0,
  "profile": "aggressive"
}
```

5. Click "Save"

### 4.2 Run Test

1. Click "Test" button
2. Wait 30-60 seconds (first run includes cold start)
3. Check results in "Execution results" section

### 4.3 Expected Results

**First Run (Cold Start):**
- Duration: 30-60 seconds
- Status: 200
- Body contains:
  - `"cached": false`
  - `"source": "lambda"`
  - `"results"` with scan data
  - `"execution_time"` in seconds

**Second Run (Warm + Redis Cache):**
- Duration: 1-2 seconds
- Status: 200
- Body contains:
  - `"cached": true`
  - `"source": "redis"`
  - Same results as first run

### 4.4 Check Logs

1. Go to "Monitor" tab
2. Click "View CloudWatch logs"
3. Click the latest log stream
4. Look for:
   - `[LAMBDA] Connected to Redis Cloud` ✅
   - `[LAMBDA] Starting scan with config` ✅
   - `[LAMBDA] Scan completed in X.XXs` ✅
   - `[LAMBDA] Found X results` ✅

---

## 🔗 Step 5: Integrate with Render

### 5.1 Get AWS Credentials

1. Go to AWS Console → IAM
2. Click "Users" → "Create user"
3. User name: `technic-lambda-user`
4. Click "Next"
5. Attach policies:
   - `AWSLambdaFullAccess`
6. Click "Create user"
7. Click on the user
8. Go to "Security credentials" tab
9. Click "Create access key"
10. Choose "Application running outside AWS"
11. **Save the Access Key ID and Secret Access Key**

### 5.2 Add to Render Environment

1. Go to Render Dashboard
2. Select your `technic` service
3. Go to "Environment" tab
4. Add these variables:

```
USE_LAMBDA = true
LAMBDA_FUNCTION_NAME = technic-scanner
AWS_REGION = us-east-1
AWS_ACCESS_KEY_ID = [Your AWS Access Key ID]
AWS_SECRET_ACCESS_KEY = [Your AWS Secret Access Key]
REDIS_URL = redis://:ytvZ1OVXoGV4OenJH3GJYkDLeg2emqad@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0
```

5. Click "Save Changes" (will trigger redeploy)

### 5.3 Update Render Start Command

1. Go to "Settings" tab
2. Find "Build & Deploy" section
3. Update "Start Command" to:
   ```
   python api_hybrid.py
   ```
4. Click "Save Changes"

### 5.4 Deploy to Render

```bash
# Commit and push changes
git add api_hybrid.py lambda_scanner.py requirements_lambda.txt
git commit -m "Add Lambda + Redis hybrid architecture"
git push origin main
```

Render will auto-deploy in ~5 minutes.

---

## ✅ Step 6: Test End-to-End

### 6.1 Test Health Endpoint

```bash
curl https://technic-m5vn.onrender.com/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "lambda_available": true,
  "redis_available": true,
  "lambda_function": "technic-scanner"
}
```

### 6.2 Test Scan Endpoint (First Call - Lambda)

```bash
curl -X POST https://technic-m5vn.onrender.com/scan \
  -H "Content-Type: application/json" \
  -d '{
    "sectors": ["Technology"],
    "max_symbols": 5,
    "min_tech_rating": 10.0
  }'
```

**Expected:**
- Response time: 20-40 seconds
- `"source": "lambda"`
- `"cached": false`
- Results with 5 technology stocks

### 6.3 Test Scan Endpoint (Second Call - Redis Cache)

```bash
# Run the same command again immediately
curl -X POST https://technic-m5vn.onrender.com/scan \
  -H "Content-Type: application/json" \
  -d '{
    "sectors": ["Technology"],
    "max_symbols": 5,
    "min_tech_rating": 10.0
  }'
```

**Expected:**
- Response time: <2 seconds
- `"source": "redis"`
- `"cached": true`
- Same results as first call

---

## 📊 Step 7: Monitor Performance

### 7.1 Check Lambda Metrics

1. Go to AWS Console → Lambda → technic-scanner
2. Click "Monitor" tab
3. Check metrics:
   - Invocations: Should increase with each scan
   - Duration: Should be 20-40s average
   - Errors: Should be 0%
   - Throttles: Should be 0

### 7.2 Check CloudWatch Logs

1. Click "View CloudWatch logs"
2. Monitor for:
   - Redis connection success
   - Scan completion messages
   - Any errors or warnings

### 7.3 Check Costs

1. Go to AWS Console → Billing
2. Click "Bills"
3. Check Lambda costs:
   - Should be $0 in free tier
   - After free tier: ~$0.01-0.03 per scan

### 7.4 Monitor Cache Hit Rate

```bash
# Get cache statistics
curl https://technic-m5vn.onrender.com/cache/stats
```

**Target Metrics:**
- Cache hit rate: >70%
- Average response time: <10s
- Lambda invocations: <30% of total requests

---

## 🎯 Success Checklist

- [ ] Deployment package created (technic-scanner.zip)
- [ ] Lambda function created in AWS
- [ ] ZIP uploaded successfully
- [ ] Memory set to 10GB
- [ ] Timeout set to 15 minutes
- [ ] REDIS_URL environment variable added
- [ ] POLYGON_API_KEY environment variable added
- [ ] Test event created and passed
- [ ] Redis connection confirmed in logs
- [ ] Scan results returned successfully
- [ ] Second test shows cache hit (<2s)
- [ ] AWS credentials created for Render
- [ ] Render environment variables updated
- [ ] api_hybrid.py deployed to Render
- [ ] Health endpoint shows lambda_available: true
- [ ] First scan uses Lambda (20-40s)
- [ ] Second scan uses Redis cache (<2s)
- [ ] Cache hit rate >70%
- [ ] No errors in CloudWatch logs
- [ ] Costs within expected range (<$10/month)

---

## 🐛 Troubleshooting

### Issue: ZIP file too large (>50MB)

**Solution:** Use S3 upload method (Option B above)

### Issue: Lambda timeout

**Solution:** 
1. Reduce max_symbols in test event
2. Check Polygon API is responding
3. Verify network connectivity

### Issue: Redis connection failed

**Solution:**
1. Verify REDIS_URL is correct
2. Test connection locally:
   ```python
   import redis
   r = redis.from_url("redis://:ytvZ1OVXoGV4OenJH3GJYkDLeg2emqad@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0")
   print(r.ping())  # Should print True
   ```
3. Check Redis Cloud dashboard

### Issue: Lambda out of memory

**Solution:**
1. Increase memory to 12GB or 14GB
2. Check CloudWatch logs for memory usage
3. Optimize code if needed

### Issue: High costs

**Solution:**
1. Check invocation count in CloudWatch
2. Verify cache is working (should reduce Lambda calls)
3. Increase cache TTL if needed

---

## 📞 Need Help?

If you encounter any issues:

1. **Check CloudWatch Logs** - Most issues show up here
2. **Verify Environment Variables** - Common source of errors
3. **Test Lambda Independently** - Isolate the problem
4. **Check Render Logs** - For integration issues

---

## 🎉 What You'll Achieve

**Performance:**
- 3-6x faster scans on average
- <2s for cached requests (70-85% of traffic)
- 20-40s for uncached requests (15-30% of traffic)

**Cost:**
- Only $10-17/month additional
- $3-5/month with free tier (first year)
- Pay only for what you use

**Reliability:**
- AWS 99.99% uptime SLA
- Auto-scaling to handle any load
- No cold start issues with proper configuration

**User Experience:**
- Much faster response times
- Better app performance
- Happier users!

---

**Current Status:** Waiting for deployment package to complete...

Once `deploy_lambda.ps1` finishes, proceed to Step 2 (Upload to AWS Lambda).
