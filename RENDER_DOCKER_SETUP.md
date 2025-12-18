# Render Integration for Docker Setup - Simple Guide

## Your Current Setup

You're using Docker with `bash start.sh` which runs:
```bash
python -m uvicorn technic_v4.api_server:app --host 0.0.0.0 --port "$PORT"
```

**Good news:** You don't need to change anything in the Docker Command field!

---

## Step 1: Add Environment Variables to Render

1. **Go to your Render dashboard:** https://dashboard.render.com
2. **Select your `technic` service**
3. **Click "Environment" in the left sidebar**
4. **Add these 5 environment variables:**

### Variables to Add:

```
Variable Name: USE_LAMBDA
Value: true
```

```
Variable Name: LAMBDA_FUNCTION_NAME  
Value: technic-scanner
```

```
Variable Name: AWS_REGION
Value: us-east-1
```

```
Variable Name: AWS_ACCESS_KEY_ID
Value: AKIASQLYU66Q3JC42XVC
```

```
Variable Name: AWS_SECRET_ACCESS_KEY
Value: Tcefr9XVk2PbbZj8/4F/7yrH8F89fDfHOMBenN68
```

5. **Click "Save Changes"**

This will trigger an automatic redeploy (~5-7 minutes).

---

## Step 2: Wait for Deployment

After saving the environment variables:

1. **Go to "Logs" tab** to watch the deployment
2. **Wait for these messages:**
   ```
   Building...
   Deploying...
   Starting service with 'bash start.sh'
   ```
3. **Deployment takes 5-7 minutes**

---

## Step 3: Test the Integration

Once deployment completes, test your API:

### Test 1: Health Check

```bash
curl https://technic-m5vn.onrender.com/health
```

**Expected:**
```json
{
  "status": "ok"
}
```

### Test 2: Run a Scan

```bash
curl -X POST https://technic-m5vn.onrender.com/v1/scan \
  -H "Content-Type: application/json" \
  -d "{\"max_symbols\": 5, \"sectors\": [\"Technology\"]}"
```

**Expected:**
- Takes 20-40 seconds (first time - Lambda)
- Returns scan results with symbols

### Test 3: Run Same Scan Again

Run the exact same command again immediately:

```bash
curl -X POST https://technic-m5vn.onrender.com/v1/scan \
  -H "Content-Type: application/json" \
  -d "{\"max_symbols\": 5, \"sectors\": [\"Technology\"]}"
```

**Expected:**
- Takes <2 seconds (Redis cache!)
- Returns same results much faster

---

## ✅ Success Criteria

Your integration is working when:

- ✅ Health endpoint responds
- ✅ First scan takes 20-40 seconds
- ✅ Second scan takes <2 seconds
- ✅ No errors in Render logs

---

## 🐛 Troubleshooting

### If deployment fails:

1. Check Render logs for specific errors
2. Verify all 5 environment variables are set correctly
3. Make sure there are no typos in the variable names

### If scans are slow:

1. Check if Redis caching is working (second scan should be fast)
2. Verify Lambda is being called (check AWS CloudWatch logs)
3. Ensure AWS credentials are correct

---

## 📊 What You'll Get

After successful integration:

**Performance:**
- First scan: 20-40s (Lambda processing)
- Cached scan: <2s (Redis cache)
- Average: ~10s (6-12x faster!)

**Cost:**
- Additional: ~$10-15/month for Lambda
- Total: ~$185-192/month

**Benefits:**
- Much faster API responses
- Automatic caching
- Production-ready scalability

---

## 🎯 Summary

**What to do:**
1. Add 5 environment variables in Render (Step 1 above)
2. Wait for redeploy (~5-7 minutes)
3. Test with the curl commands (Step 3 above)

**What NOT to do:**
- ❌ Don't change the Docker Command (keep `bash start.sh`)
- ❌ Don't modify start.sh
- ❌ Don't change any code files

**The environment variables are all you need!**

---

## 📝 Your AWS Credentials (Save These!)

- **Access Key ID:** `AKIASQLYU66Q3JC42XVC`
- **Secret Access Key:** `Tcefr9XVk2PbbZj8/4F/7yrH8F89fDfHOMBenN68`

⚠️ Keep these secure! They provide access to your Lambda function.

---

**Ready? Go to Render and add those 5 environment variables!**
