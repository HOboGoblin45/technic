# Lambda Testing & Render Integration Guide

## ✅ Current Status

- ✅ Lambda function created on AWS
- ✅ Deployment package uploaded (113.91 MB)
- ✅ Lambda configured (memory, timeout, environment variables)
- ⏳ **Next: Test Lambda & Integrate with Render**

---

## 🧪 Step 1: Test Lambda Function

### Test in AWS Console

1. **Go to Lambda Console**
   - Navigate to: https://console.aws.amazon.com/lambda
   - Select function: `technic-scanner`

2. **Create Test Event**
   - Click "Test" tab
   - Click "Create new event"
   - Event name: `test-tech-scan`
   - Event JSON:
   ```json
   {
     "sectors": ["Technology"],
     "max_symbols": 5,
     "min_tech_rating": 10.0,
     "profile": "aggressive"
   }
   ```
   - Click "Save"

3. **Run Test**
   - Click "Test" button
   - Wait 30-60 seconds (first run includes cold start)

4. **Check Results**

   **Expected Response:**
   ```json
   {
     "statusCode": 200,
     "body": {
       "cached": false,
       "source": "lambda",
       "results": {
         "symbols": [...],
         "status": "...",
         "metrics": {...}
       },
       "execution_time": 45.2,
       "lambda_info": {
         "memory_limit": 10240,
         "memory_used": 8500,
         "time_remaining": 850000
       }
     }
   }
   ```

5. **Run Test Again (Check Redis Cache)**
   - Click "Test" button again immediately
   - Should complete in 1-2 seconds
   - Response should show:
     ```json
     {
       "cached": true,
       "source": "redis"
     }
     ```

### Check CloudWatch Logs

1. Go to "Monitor" tab
2. Click "View CloudWatch logs"
3. Click latest log stream
4. Look for these messages:

   **✅ Success Indicators:**
   ```
   [LAMBDA] Function: technic-scanner
   [LAMBDA] Memory: 10240MB
   [LAMBDA] Connected to Redis Cloud
   [LAMBDA] Starting scan with config
   [LAMBDA] Scan completed in X.XXs
   [LAMBDA] Found X results
   [LAMBDA] Cached result for 300s
   ```

   **❌ Error Indicators:**
   ```
   Failed to connect to Redis
   Timeout error
   Memory error
   ```

### Troubleshooting

**If Redis connection fails:**
1. Verify REDIS_URL environment variable is set correctly
2. Check Redis Cloud dashboard - ensure instance is running
3. Test Redis connection locally:
   ```python
   import redis
   r = redis.from_url("redis://:ytvZ1OVXoGV4OenJH3GJYkDLeg2emqad@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0")
   print(r.ping())  # Should print True
   ```

**If Lambda times out:**
1. Check Polygon API key is valid
2. Reduce max_symbols in test event
3. Check CloudWatch logs for specific errors

**If out of memory:**
1. Increase memory to 12GB or 14GB
2. Check CloudWatch logs for memory usage

---

## 🔗 Step 2: Integrate with Render

### 2.1 Get AWS Credentials

If you don't have AWS credentials yet:

1. **Go to AWS IAM Console**
   - Navigate to: https://console.aws.amazon.com/iam

2. **Create IAM User**
   - Click "Users" → "Create user"
   - User name: `technic-lambda-user`
   - Click "Next"

3. **Set Permissions**
   - Click "Attach policies directly"
   - Search and select: `AWSLambdaFullAccess`
   - Click "Next" → "Create user"

4. **Create Access Key**
   - Click on the user you just created
   - Go to "Security credentials" tab
   - Click "Create access key"
   - Choose "Application running outside AWS"
   - Click "Create access key"
   - **Save the Access Key ID and Secret Access Key** (you won't see them again!)

### 2.2 Add Environment Variables to Render

1. **Go to Render Dashboard**
   - Navigate to: https://dashboard.render.com
   - Select your `technic` service

2. **Add Environment Variables**
   - Go to "Environment" tab
   - Click "Add Environment Variable"
   - Add these variables:

   ```
   USE_LAMBDA = true
   LAMBDA_FUNCTION_NAME = technic-scanner
   AWS_REGION = us-east-1
   AWS_ACCESS_KEY_ID = [Your AWS Access Key ID]
   AWS_SECRET_ACCESS_KEY = [Your AWS Secret Access Key]
   REDIS_URL = redis://:ytvZ1OVXoGV4OenJH3GJYkDLeg2emqad@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0
   ```

3. **Save Changes**
   - Click "Save Changes"
   - This will trigger a redeploy (takes ~5 minutes)

### 2.3 Update Render Start Command

1. **Go to Settings**
   - Click "Settings" tab
   - Scroll to "Build & Deploy" section

2. **Update Start Command**
   - Change from: `python api.py`
   - To: `python api_hybrid.py`
   - Click "Save Changes"

### 2.4 Deploy to Render

**Option A: Auto-deploy (if connected to GitHub)**

1. Push code to GitHub:
   ```bash
   git add api_hybrid.py lambda_scanner.py requirements_lambda.txt
   git commit -m "Add Lambda + Redis hybrid architecture"
   git push origin main
   ```

2. Render will auto-deploy in ~5 minutes

**Option B: Manual deploy**

1. Go to Render Dashboard
2. Click "Manual Deploy" → "Deploy latest commit"
3. Wait for deployment to complete

### 2.5 Verify Deployment

Wait for Render deployment to complete, then check the logs:

1. Go to "Logs" tab in Render
2. Look for:
   ```
   Starting service with 'python api_hybrid.py'
   Lambda client initialized
   Redis client initialized
   ```

---

## ✅ Step 3: Test End-to-End

### 3.1 Test Health Endpoint

```bash
curl https://technic-m5vn.onrender.com/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "lambda_available": true,
  "redis_available": true,
  "lambda_function": "technic-scanner",
  "redis_connected": true
}
```

**If lambda_available is false:**
- Check AWS credentials in Render environment
- Verify Lambda function name is correct
- Check Render logs for errors

**If redis_available is false:**
- Check REDIS_URL in Render environment
- Verify Redis Cloud instance is running
- Check Render logs for connection errors

### 3.2 Test Scan Endpoint (First Call - Lambda)

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
- Response includes:
  ```json
  {
    "cached": false,
    "source": "lambda",
    "results": {
      "symbols": [...],
      "status": "...",
      "metrics": {...}
    },
    "execution_time": 35.2
  }
  ```

### 3.3 Test Scan Endpoint (Second Call - Redis Cache)

Run the same command again immediately:

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
- Response time: <2 seconds
- Response includes:
  ```json
  {
    "cached": true,
    "source": "redis",
    "results": {...},
    "execution_time": 1.5
  }
  ```

### 3.4 Test Different Configurations

```bash
# Test different sector
curl -X POST https://technic-m5vn.onrender.com/scan \
  -H "Content-Type: application/json" \
  -d '{
    "sectors": ["Healthcare"],
    "max_symbols": 5,
    "min_tech_rating": 10.0
  }'

# Test different profile
curl -X POST https://technic-m5vn.onrender.com/scan \
  -H "Content-Type: application/json" \
  -d '{
    "sectors": ["Technology"],
    "max_symbols": 5,
    "profile": "conservative"
  }'
```

### 3.5 Test Force Lambda (Bypass Cache)

```bash
curl -X POST https://technic-m5vn.onrender.com/scan \
  -H "Content-Type: application/json" \
  -d '{
    "sectors": ["Technology"],
    "max_symbols": 5,
    "force_lambda": true
  }'
```

Should always use Lambda, even if cached.

---

## 📊 Step 4: Monitor Performance

### 4.1 Check Lambda Metrics

1. **Go to Lambda Console**
   - Navigate to: https://console.aws.amazon.com/lambda
   - Select function: `technic-scanner`
   - Click "Monitor" tab

2. **Check Metrics**
   - **Invocations:** Should increase with each scan
   - **Duration:** Should be 20-40s average
   - **Errors:** Should be 0%
   - **Throttles:** Should be 0
   - **Concurrent executions:** Should be 1-2

3. **Check CloudWatch Logs**
   - Click "View CloudWatch logs"
   - Monitor for errors or warnings
   - Verify Redis connections are successful

### 4.2 Check Cache Statistics

```bash
curl https://technic-m5vn.onrender.com/cache/stats
```

**Expected Response:**
```json
{
  "cache_hit_rate": 0.75,
  "total_requests": 100,
  "cache_hits": 75,
  "cache_misses": 25,
  "average_response_time": 8.5
}
```

**Target Metrics:**
- Cache hit rate: >70%
- Average response time: <10s
- Lambda invocations: <30% of total requests

### 4.3 Check AWS Costs

1. **Go to AWS Billing Console**
   - Navigate to: https://console.aws.amazon.com/billing

2. **Check Lambda Costs**
   - Click "Bills"
   - Filter by service: "AWS Lambda"
   - Check current month charges

**Expected Costs (Alpha/Beta):**
- Compute: ~$3-5/month
- Requests: ~$0.00
- **Total: ~$3-5/month**

### 4.4 Set Up Cost Alerts

1. **Go to AWS Budgets**
   - Navigate to: https://console.aws.amazon.com/billing/home#/budgets

2. **Create Budget**
   - Click "Create budget"
   - Budget type: "Cost budget"
   - Name: "Lambda Monthly Budget"
   - Amount: $50/month
   - Alert threshold: 80% ($40)
   - Email: your_email@example.com
   - Click "Create budget"

---

## 🎯 Success Criteria

### ✅ Lambda Working
- [ ] Test event passes in AWS Console
- [ ] First run takes 30-60s
- [ ] Second run takes 1-2s (Redis cache)
- [ ] CloudWatch logs show Redis connection
- [ ] No errors in CloudWatch logs

### ✅ Render Integration Working
- [ ] Environment variables added
- [ ] Start command updated
- [ ] Deployment successful
- [ ] Health endpoint shows lambda_available: true
- [ ] Health endpoint shows redis_available: true

### ✅ End-to-End Working
- [ ] First scan uses Lambda (20-40s)
- [ ] Second scan uses Redis cache (<2s)
- [ ] Different configurations work
- [ ] Force Lambda works
- [ ] Cache hit rate >70%

### ✅ Performance Targets Met
- [ ] Cached scans: <2s
- [ ] Uncached scans: 20-40s
- [ ] Average response time: <10s
- [ ] Cache hit rate: >70%
- [ ] Lambda costs: <$10/month

---

## 🐛 Common Issues & Solutions

### Issue 1: Lambda Not Available in Render

**Symptoms:**
- Health endpoint shows `lambda_available: false`
- Scans fail with "Lambda not configured"

**Solutions:**
1. Check AWS credentials in Render environment
2. Verify Lambda function name is `technic-scanner`
3. Check AWS region is `us-east-1`
4. Test AWS credentials locally:
   ```python
   import boto3
   client = boto3.client('lambda', region_name='us-east-1')
   response = client.list_functions()
   print(response)
   ```

### Issue 2: Redis Not Available

**Symptoms:**
- Health endpoint shows `redis_available: false`
- All scans use Lambda (no caching)

**Solutions:**
1. Check REDIS_URL in Render environment
2. Verify Redis Cloud instance is running
3. Test Redis connection locally
4. Check Redis Cloud firewall rules

### Issue 3: Slow Response Times

**Symptoms:**
- Scans take >60s
- Cache doesn't seem to work

**Solutions:**
1. Check Lambda CloudWatch logs for errors
2. Verify cache TTL is set correctly (300s)
3. Check if cache keys are being generated correctly
4. Monitor Lambda memory usage

### Issue 4: High Lambda Costs

**Symptoms:**
- Monthly bill >$50
- More invocations than expected

**Solutions:**
1. Check cache hit rate (should be >70%)
2. Increase cache TTL if needed
3. Verify cache is working properly
4. Check for unnecessary Lambda invocations

---

## 📈 Performance Comparison

### Before (Render Only)
- **Uncached scan:** 60-120s
- **Cached scan:** N/A (no cache)
- **Average:** 60-120s
- **Cost:** $175/month

### After (Lambda + Redis)
- **First scan:** 20-40s (2-3x faster)
- **Cached scan:** <2s (30-60x faster)
- **Average:** ~10s (6-12x faster)
- **Cost:** $185-192/month (+$10-17)

### ROI Analysis
- **Additional cost:** $10-17/month
- **Performance improvement:** 6-12x faster
- **User experience:** Dramatically better
- **Scalability:** Can handle 10x more users
- **Reliability:** 99.99% uptime (AWS SLA)

---

## 🎉 Next Steps After Integration

### Immediate (This Week)
1. ✅ Monitor performance for 24-48 hours
2. ✅ Check cache hit rate
3. ✅ Verify costs are within budget
4. ✅ Fix any issues that arise

### Short Term (Next 2 Weeks)
1. Add loading indicators to Flutter app
2. Add progress tracking for scans
3. Implement error handling improvements
4. Add cache status UI

### Medium Term (Next Month)
1. Optimize cache TTL based on usage patterns
2. Add Lambda provisioned concurrency if needed
3. Implement advanced monitoring
4. Plan for scaling

### Long Term (Next Quarter)
1. Evaluate AWS migration for full stack
2. Consider mobile app development
3. Add enterprise features
4. Plan for beta launch

---

## 📞 Support

**If you encounter issues:**

1. **Check CloudWatch Logs** - Most issues show up here
2. **Check Render Logs** - For integration issues
3. **Test Lambda Independently** - Isolate the problem
4. **Verify Environment Variables** - Common source of errors
5. **Review this guide** - Step-by-step troubleshooting

**Common issues are usually:**
- Environment variables not set correctly
- AWS credentials invalid or expired
- Redis connection string incorrect
- Lambda timeout too short
- Memory too low

---

## ✅ Completion Checklist

- [ ] Lambda tested in AWS Console
- [ ] Redis cache working (second test <2s)
- [ ] CloudWatch logs reviewed
- [ ] AWS credentials created
- [ ] Render environment variables added
- [ ] Render start command updated
- [ ] Code deployed to Render
- [ ] Health endpoint tested
- [ ] First scan tested (Lambda)
- [ ] Second scan tested (Redis cache)
- [ ] Different configurations tested
- [ ] Performance metrics checked
- [ ] Cost alerts configured
- [ ] Documentation reviewed

**Once all items are checked, your hybrid Lambda + Redis architecture is fully deployed and operational!** 🎉

---

**Current Status:** Lambda uploaded and configured ✅  
**Next Step:** Test Lambda in AWS Console, then integrate with Render
