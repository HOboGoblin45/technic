# Render Integration - Step-by-Step Guide

## ✅ Step 1: AWS Credentials Created

Your AWS IAM user has been created successfully!

**AWS Credentials:**
- **Access Key ID:** `AKIASQLYU66Q3JC42XVC`
- **Secret Access Key:** `Tcefr9XVk2PbbZj8/4F/7yrH8F89fDfHOMBenN68`

⚠️ **IMPORTANT:** Keep these credentials secure! They provide access to your Lambda function.

---

## 🎯 Step 2: Add Environment Variables to Render

### Instructions:

1. **Go to Render Dashboard:**
   - Open: https://dashboard.render.com
   - Select your `technic` service

2. **Navigate to Environment Tab:**
   - Click on "Environment" in the left sidebar

3. **Add These Environment Variables:**

Click "Add Environment Variable" for each of these:

```
Variable Name: USE_LAMBDA
Value: true

Variable Name: LAMBDA_FUNCTION_NAME
Value: technic-scanner

Variable Name: AWS_REGION
Value: us-east-1

Variable Name: AWS_ACCESS_KEY_ID
Value: AKIASQLYU66Q3JC42XVC

Variable Name: AWS_SECRET_ACCESS_KEY
Value: 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
```

4. **Click "Save Changes"**
   - This will trigger an automatic redeploy (~5 minutes)

---

## 🔧 Step 3: Update Render Start Command

### Instructions:

1. **Go to Settings Tab:**
   - Click on "Settings" in the left sidebar

2. **Find "Build & Deploy" Section:**
   - Scroll down to find the "Start Command" field

3. **Update Start Command:**
   - **Current:** `python api.py`
   - **Change to:** `python api_hybrid.py`

4. **Click "Save Changes"**
   - If you already saved environment variables, this will trigger another redeploy
   - If not, it will redeploy when you save the environment variables

---

## ⏱️ Step 4: Wait for Deployment

After saving changes, Render will redeploy your service:

1. **Go to "Logs" tab** to monitor the deployment
2. **Wait for these messages:**
   ```
   Starting service with 'python api_hybrid.py'
   Lambda client initialized
   Redis client initialized
   ```
3. **Deployment typically takes 5-7 minutes**

---

## ✅ Step 5: Test the Integration

Once deployment is complete, test these endpoints:

### Test 1: Health Check

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

### Test 2: First Scan (Uses Lambda)

```bash
curl -X POST https://technic-m5vn.onrender.com/scan \
  -H "Content-Type: application/json" \
  -d "{\"sectors\": [\"Technology\"], \"max_symbols\": 5}"
```

**Expected:**
- Takes 20-40 seconds
- Response includes: `"source": "lambda"` and `"cached": false`

### Test 3: Second Scan (Uses Redis Cache)

Run the same command again immediately:

```bash
curl -X POST https://technic-m5vn.onrender.com/scan \
  -H "Content-Type: application/json" \
  -d "{\"sectors\": [\"Technology\"], \"max_symbols\": 5}"
```

**Expected:**
- Takes <2 seconds
- Response includes: `"source": "redis"` and `"cached": true`

---

## 🎉 Success Criteria

Your integration is successful when:

- ✅ Health endpoint shows `lambda_available: true`
- ✅ Health endpoint shows `redis_available: true`
- ✅ First scan takes 20-40 seconds (Lambda)
- ✅ Second scan takes <2 seconds (Redis cache)
- ✅ No errors in Render logs

---

## 🐛 Troubleshooting

### If `lambda_available: false`

**Check:**
1. AWS credentials are correct in Render environment
2. Lambda function name is exactly `technic-scanner`
3. AWS region is `us-east-1`
4. Render logs for specific error messages

### If `redis_available: false`

**Check:**
1. REDIS_URL environment variable is set in Render
2. Redis Cloud instance is running
3. Render logs for connection errors

### If scans are slow

**Check:**
1. CloudWatch logs for Lambda errors
2. Lambda memory/timeout settings
3. Cache is working (second scan should be fast)

---

## 📊 What You'll Get

After successful integration:

**Performance Improvements:**
- First scan: 20-40s (2-3x faster than before)
- Cached scan: <2s (30-60x faster!)
- Average response: ~10s (6-12x faster)

**Cost:**
- Additional: ~$10-15/month
- Total: ~$185-192/month

**User Experience:**
- Dramatically faster responses
- Seamless caching
- Production-ready reliability

---

## 📝 Next Steps After Integration

Once integration is complete and tested:

1. ✅ Monitor performance for 24-48 hours
2. ✅ Check cache hit rate (should be >70%)
3. ✅ Verify AWS costs are within budget
4. ✅ Update Flutter app to use the faster API
5. ✅ Add loading indicators for better UX

---

## 🎯 Current Status

- ✅ Lambda deployed and tested
- ✅ AWS credentials created
- ⏳ **Next:** Add environment variables to Render (Step 2 above)

---

**Ready to proceed? Follow Step 2 above to add the environment variables to Render!**
