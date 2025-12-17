# Hybrid Approach Deployment Guide

Complete step-by-step guide to deploy Render + AWS Lambda hybrid architecture

---

## 📋 Prerequisites

### What You Need
- ✅ AWS Account (free tier works)
- ✅ Render Pro Plus account (you have this)
- ✅ Redis Cloud 12GB (you have this)
- ✅ GitHub repository with your code
- ✅ AWS CLI installed (optional but recommended)

### Files Created
- ✅ `lambda_scanner.py` - Lambda function code
- ✅ `api_hybrid.py` - Hybrid API for Render
- ✅ `HYBRID_DEPLOYMENT_GUIDE.md` - This guide

---

## 🚀 Phase 1: AWS Lambda Setup (Day 1-2)

### Step 1: Create AWS Account

1. Go to https://aws.amazon.com
2. Click "Create an AWS Account"
3. Follow the signup process
4. Add payment method (won't be charged during free tier)
5. Verify your identity

**Cost: $0 (free tier includes 1M Lambda requests/month)**

### Step 2: Create IAM User for Lambda

1. Go to AWS Console → IAM
2. Click "Users" → "Add User"
3. User name: `technic-lambda-user`
4. Access type: ✅ Programmatic access
5. Permissions: Attach existing policy → `AWSLambdaFullAccess`
6. Create user
7. **Save Access Key ID and Secret Access Key** (you'll need these)

### Step 3: Install AWS CLI (Optional)

**Windows:**
```powershell
# Download and install from:
https://awscli.amazonaws.com/AWSCLIV2.msi

# Configure
aws configure
# Enter your Access Key ID
# Enter your Secret Access Key
# Region: us-east-1
# Output format: json
```

**Mac:**
```bash
brew install awscli
aws configure
```

### Step 4: Create Lambda Function

#### Option A: Using AWS Console (Easier)

1. Go to AWS Console → Lambda
2. Click "Create function"
3. Choose "Author from scratch"
4. Function name: `technic-scanner`
5. Runtime: Python 3.11
6. Architecture: x86_64
7. Click "Create function"

#### Option B: Using AWS CLI

```bash
# Create function
aws lambda create-function \
  --function-name technic-scanner \
  --runtime python3.11 \
  --role arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-execution-role \
  --handler lambda_scanner.lambda_handler \
  --timeout 900 \
  --memory-size 10240
```

### Step 5: Configure Lambda Settings

1. Go to your Lambda function
2. Configuration → General configuration → Edit
3. Memory: **10240 MB** (10GB)
4. Timeout: **15 minutes** (900 seconds)
5. Ephemeral storage: **512 MB** (default is fine)
6. Save

### Step 6: Add Environment Variables

1. Configuration → Environment variables → Edit
2. Add these variables:

```
REDIS_URL = redis://:ytvZ1OVXoGV4OenJH3GJYkDLeg2emqad@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0
POLYGON_API_KEY = your_polygon_api_key
```

3. Save

### Step 7: Package and Deploy Lambda Code

#### Create deployment package:

```bash
# Create a directory for the package
mkdir lambda_package
cd lambda_package

# Copy your code
cp ../lambda_scanner.py .
cp -r ../technic_v4 .

# Install dependencies
pip install -r ../requirements.txt -t .

# Create ZIP file
zip -r ../lambda_deployment.zip .

# Go back
cd ..
```

#### Upload to Lambda:

**Option A: Using Console**
1. Go to Lambda function
2. Code → Upload from → .zip file
3. Select `lambda_deployment.zip`
4. Save

**Option B: Using AWS CLI**
```bash
aws lambda update-function-code \
  --function-name technic-scanner \
  --zip-file fileb://lambda_deployment.zip
```

### Step 8: Test Lambda Function

1. Go to Lambda function → Test
2. Create new test event:

```json
{
  "sectors": ["Technology"],
  "max_symbols": 5,
  "min_tech_rating": 10.0,
  "profile": "aggressive"
}
```

3. Click "Test"
4. Check execution results
5. Should see scan results in ~30-60 seconds

**If successful, Lambda is ready!** ✅

---

## 🔧 Phase 2: Render Integration (Day 3-4)

### Step 1: Update Render Environment Variables

1. Go to Render Dashboard
2. Select your `technic` service
3. Environment → Add Environment Variable
4. Add these:

```
USE_LAMBDA = true
LAMBDA_FUNCTION_NAME = technic-scanner
AWS_REGION = us-east-1
AWS_ACCESS_KEY_ID = your_access_key_id
AWS_SECRET_ACCESS_KEY = your_secret_access_key
```

5. Save changes (will trigger redeploy)

### Step 2: Update Render Start Command

1. Go to Settings → Build & Deploy
2. Start Command: `python api_hybrid.py`
3. Save

### Step 3: Deploy Hybrid API

1. Push code to GitHub:

```bash
git add lambda_scanner.py api_hybrid.py
git commit -m "Add hybrid Lambda + Render architecture"
git push origin main
```

2. Render will auto-deploy
3. Wait for deployment to complete (~5 minutes)

### Step 4: Test Hybrid API

```bash
# Test health endpoint
curl https://technic-m5vn.onrender.com/health

# Should return:
{
  "status": "healthy",
  "lambda_available": true,
  "redis_available": true,
  "lambda_function": "technic-scanner"
}

# Test scan endpoint
curl -X POST https://technic-m5vn.onrender.com/scan \
  -H "Content-Type: application/json" \
  -d '{
    "sectors": ["Technology"],
    "max_symbols": 5,
    "min_tech_rating": 10.0
  }'

# First call: Should use Lambda (20-40s)
# Second call: Should use cache (<2s)
```

**If successful, hybrid API is working!** ✅

---

## 📊 Phase 3: Monitoring & Optimization (Day 5)

### Step 1: Set Up CloudWatch Monitoring

1. Go to AWS Console → CloudWatch
2. Dashboards → Create dashboard
3. Name: `technic-scanner-metrics`
4. Add widgets:
   - Lambda invocations
   - Lambda duration
   - Lambda errors
   - Lambda concurrent executions

### Step 2: Set Up Cost Alerts

1. Go to AWS Console → Billing
2. Budgets → Create budget
3. Budget type: Cost budget
4. Amount: $50/month
5. Alert threshold: 80% ($40)
6. Email: your_email@example.com
7. Create budget

### Step 3: Monitor Performance

**Check Lambda metrics:**
```bash
# Get Lambda metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Duration \
  --dimensions Name=FunctionName,Value=technic-scanner \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z \
  --period 3600 \
  --statistics Average
```

**Check API metrics:**
```bash
# Get cache stats
curl https://technic-m5vn.onrender.com/cache/stats

# Get Lambda info
curl https://technic-m5vn.onrender.com/lambda/info
```

### Step 4: Optimize Lambda Performance

**If cold starts are slow:**

1. Enable Provisioned Concurrency:
   - Go to Lambda → Configuration → Concurrency
   - Add provisioned concurrency: 1
   - Cost: ~$5/month
   - Eliminates cold starts

2. Optimize package size:
   - Remove unused dependencies
   - Use Lambda layers for common libraries
   - Reduce deployment package size

**If execution is slow:**

1. Increase memory (more memory = more CPU):
   - Try 12GB or 14GB
   - Monitor performance vs cost

2. Optimize code:
   - Profile slow functions
   - Add parallel processing
   - Optimize database queries

---

## 🧪 Phase 4: Testing (Week 2)

### Test Scenarios

#### Test 1: Cache Hit
```bash
# First scan (cache miss)
time curl -X POST https://technic-m5vn.onrender.com/scan \
  -H "Content-Type: application/json" \
  -d '{"sectors": ["Technology"], "max_symbols": 10}'

# Expected: 20-40s, source: "lambda"

# Second scan (cache hit)
time curl -X POST https://technic-m5vn.onrender.com/scan \
  -H "Content-Type: application/json" \
  -d '{"sectors": ["Technology"], "max_symbols": 10}'

# Expected: <2s, source: "render_cache"
```

#### Test 2: Different Configurations
```bash
# Test different sectors
curl -X POST https://technic-m5vn.onrender.com/scan \
  -H "Content-Type: application/json" \
  -d '{"sectors": ["Healthcare"], "max_symbols": 10}'

# Test different profiles
curl -X POST https://technic-m5vn.onrender.com/scan \
  -H "Content-Type: application/json" \
  -d '{"profile": "conservative", "max_symbols": 10}'
```

#### Test 3: Force Lambda
```bash
# Force Lambda execution (bypass cache)
curl -X POST https://technic-m5vn.onrender.com/scan \
  -H "Content-Type: application/json" \
  -d '{"sectors": ["Technology"], "max_symbols": 10, "force_lambda": true}'

# Expected: 20-40s, source: "lambda"
```

#### Test 4: Concurrent Requests
```bash
# Run 10 concurrent scans
for i in {1..10}; do
  curl -X POST https://technic-m5vn.onrender.com/scan \
    -H "Content-Type: application/json" \
    -d '{"sectors": ["Technology"], "max_symbols": 5}' &
done
wait

# Check Lambda concurrent executions in CloudWatch
```

### Performance Benchmarks

**Target Metrics:**
- Cache hit: <2s
- Lambda execution: 20-40s
- Cache hit rate: >70%
- Lambda errors: <1%
- Cost per scan: <$0.02

**Measure:**
```bash
# Run 100 scans and measure
for i in {1..100}; do
  time curl -X POST https://technic-m5vn.onrender.com/scan \
    -H "Content-Type: application/json" \
    -d '{"sectors": ["Technology"], "max_symbols": 10}' \
    >> scan_results.txt
done

# Analyze results
grep "execution_time" scan_results.txt | \
  awk '{sum+=$2; count++} END {print "Average:", sum/count}'
```

---

## 💰 Cost Monitoring

### Daily Cost Check

```bash
# Check Lambda costs
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-02 \
  --granularity DAILY \
  --metrics BlendedCost \
  --filter file://lambda-filter.json

# lambda-filter.json:
{
  "Dimensions": {
    "Key": "SERVICE",
    "Values": ["AWS Lambda"]
  }
}
```

### Expected Costs (Alpha/Beta)

**Assumptions:**
- 1,000 scans/month
- 30% uncached (300 Lambda invocations)
- 10GB memory, 60s average execution

**Calculation:**
- Compute: 300 × 10GB × 60s × $0.0000166667 = $3.00
- Requests: 300 × $0.0000002 = $0.00006
- **Total: ~$3/month**

**With Render:**
- Render Pro Plus: $175/month
- Lambda: $3/month
- **Total: $178/month**

---

## 🐛 Troubleshooting

### Issue 1: Lambda Timeout

**Symptoms:**
- Scans fail after 15 minutes
- Error: "Task timed out after 900.00 seconds"

**Solutions:**
1. Reduce `max_symbols` in scan config
2. Optimize scanner code
3. Increase timeout (max 15 minutes)
4. Split into multiple Lambda calls

### Issue 2: Lambda Out of Memory

**Symptoms:**
- Error: "Runtime exited with error: signal: killed"
- Memory usage near 10GB

**Solutions:**
1. Increase memory to 12GB or 14GB
2. Optimize memory usage in code
3. Process symbols in batches
4. Clear variables after use

### Issue 3: Lambda Cold Start Slow

**Symptoms:**
- First invocation takes 10-20s
- Subsequent invocations are fast

**Solutions:**
1. Enable Provisioned Concurrency (1 instance)
2. Reduce deployment package size
3. Use Lambda layers for dependencies
4. Keep Lambda warm with scheduled pings

### Issue 4: High Lambda Costs

**Symptoms:**
- Monthly bill >$50
- More invocations than expected

**Solutions:**
1. Increase cache TTL (5 min → 15 min)
2. Reduce memory if possible
3. Optimize execution time
4. Check for unnecessary invocations

### Issue 5: Redis Connection Errors

**Symptoms:**
- Error: "Connection refused"
- Cache not working

**Solutions:**
1. Check REDIS_URL is correct
2. Verify Redis Cloud is running
3. Check network connectivity
4. Increase connection timeout

---

## 📈 Scaling Strategy

### Current (Alpha/Beta)
- Render Pro Plus: 4 CPU, 8GB RAM
- Lambda: 10GB memory, on-demand
- Redis: 12GB
- **Capacity: ~10,000 scans/month**

### Growth Phase 1 (1,000-5,000 users)
- Keep Render Pro Plus
- Increase Lambda memory to 12GB
- Add Lambda Provisioned Concurrency (2 instances)
- **Cost: $190-220/month**
- **Capacity: ~50,000 scans/month**

### Growth Phase 2 (5,000-20,000 users)
- Upgrade to Render Pro Max (16GB RAM)
- Increase Lambda memory to 14GB
- Add Lambda Provisioned Concurrency (5 instances)
- Upgrade Redis to 30GB
- **Cost: $300-400/month**
- **Capacity: ~200,000 scans/month**

### Growth Phase 3 (20,000+ users)
- Consider full AWS migration
- Use EC2 Auto Scaling
- Use ElastiCache Redis
- Use RDS for persistent data
- **Cost: $500-1000/month**
- **Capacity: Unlimited**

---

## ✅ Deployment Checklist

### Pre-Deployment
- [ ] AWS account created
- [ ] IAM user created with Lambda permissions
- [ ] AWS CLI configured (optional)
- [ ] Lambda function created
- [ ] Lambda configured (10GB memory, 15 min timeout)
- [ ] Environment variables set in Lambda
- [ ] Lambda code deployed and tested

### Render Integration
- [ ] Environment variables added to Render
- [ ] Start command updated to `python api_hybrid.py`
- [ ] Code pushed to GitHub
- [ ] Render deployment successful
- [ ] Health endpoint returns `lambda_available: true`

### Testing
- [ ] Cache hit test passed (<2s)
- [ ] Lambda execution test passed (20-40s)
- [ ] Different configurations tested
- [ ] Concurrent requests tested
- [ ] Error handling tested

### Monitoring
- [ ] CloudWatch dashboard created
- [ ] Cost alerts configured
- [ ] Performance metrics tracked
- [ ] Error logs monitored

### Documentation
- [ ] Architecture documented
- [ ] Deployment process documented
- [ ] Troubleshooting guide created
- [ ] Team trained (if applicable)

---

## 🎉 Success Criteria

**You'll know it's working when:**

1. ✅ Health endpoint shows Lambda available
2. ✅ First scan takes 20-40s (Lambda)
3. ✅ Second identical scan takes <2s (cache)
4. ✅ Cache hit rate >70%
5. ✅ Lambda costs <$10/month
6. ✅ No timeout errors
7. ✅ No memory errors
8. ✅ Users report fast scans

**Performance Targets:**
- Cached scans: <2s (70-85% of requests)
- Uncached scans: 20-40s (15-30% of requests)
- Average: ~10s (3x faster than Render only)
- Cost: $178-205/month (only $3-30 more)

---

## 📞 Support

**If you need help:**

1. Check CloudWatch logs for Lambda errors
2. Check Render logs for API errors
3. Test Lambda directly in AWS Console
4. Test API endpoints with curl
5. Review this guide's troubleshooting section

**Common issues are usually:**
- Environment variables not set correctly
- AWS credentials not configured
- Redis connection string incorrect
- Lambda timeout too short
- Memory too low

---

## 🚀 Next Steps After Deployment

1. **Week 1:** Monitor performance and costs
2. **Week 2:** Optimize based on data
3. **Week 3:** Start alpha testing with users
4. **Week 4:** Gather feedback and iterate

**Then:**
- Add mobile app deployment ($124)
- Add loading indicators to UI
- Implement real-time progress updates
- Plan for beta launch

---

**You're ready to deploy!** Follow this guide step-by-step and you'll have a lightning-fast hybrid architecture in 3 weeks. 🚀
