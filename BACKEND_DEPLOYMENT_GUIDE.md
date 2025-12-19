# Backend API Deployment Guide
**Date**: December 19, 2024
**Purpose**: Deploy Technic API to Render for Mobile App Integration

---

## 🎯 Overview

This guide covers deploying the Technic API backend to Render.com so the mobile app can connect to the `/v1/scan` and other endpoints.

---

## 📋 Current Status

### Mobile App Configuration
- **Base URL**: `https://technic-m5vn.onrender.com`
- **Scan Endpoint**: `/v1/scan` (POST)
- **Copilot Endpoint**: `/v1/copilot` (POST)
- **Symbol Detail**: `/v1/symbol/{ticker}` (GET)
- **Universe Stats**: `/v1/universe_stats` (GET)

### API Server
- **File**: `technic_v4/api_server.py`
- **Framework**: FastAPI
- **Endpoints**: All `/v1/*` routes implemented
- **Features**:
  - Full scanner with MERIT scores
  - Options analysis
  - Copilot AI integration
  - Symbol detail with fundamentals
  - Universe statistics

---

## 🚀 Deployment Steps

### Step 1: Verify Configuration Files

#### ✅ render.yaml (Created)
```yaml
services:
  - type: web
    name: technic-api
    env: python
    region: oregon
    plan: starter
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn technic_v4.api_server:app --host 0.0.0.0 --port $PORT
    envVars:
      - PYTHON_VERSION: 3.11.0
      - TECHNIC_API_KEY: (secret)
      - POLYGON_API_KEY: (secret)
      - FMP_API_KEY: (secret)
      - OPENAI_API_KEY: (secret)
      - REDIS_HOST: (secret)
      - REDIS_PORT: (secret)
      - REDIS_PASSWORD: (secret)
      - REDIS_DB: 0
      - TECHNIC_USE_REDIS: 1
      - TECHNIC_USE_ML_ALPHA: true
      - TECHNIC_USE_RAY: 0
    healthCheckPath: /health
```

#### ✅ requirements.txt (Exists)
All necessary dependencies are included:
- FastAPI & Uvicorn
- Pandas, NumPy, SciPy
- ML libraries (XGBoost, scikit-learn, PyTorch)
- Redis for caching
- OpenAI for Copilot
- Data providers (yfinance, etc.)

### Step 2: Commit Changes to Git

```bash
# Add the new render.yaml file
git add render.yaml

# Add this deployment guide
git add BACKEND_DEPLOYMENT_GUIDE.md

# Commit with descriptive message
git commit -m "Add Render deployment configuration for mobile app backend"

# Push to GitHub
git push origin main
```

### Step 3: Deploy to Render

#### Option A: Automatic Deployment (Recommended)
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Blueprint"
3. Connect your GitHub repository
4. Render will detect `render.yaml` and create the service
5. Configure environment variables (see Step 4)
6. Click "Apply" to deploy

#### Option B: Manual Service Creation
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: technic-api
   - **Region**: Oregon (US West)
   - **Branch**: main
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn technic_v4.api_server:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Starter ($7/month)
5. Add environment variables (see Step 4)
6. Click "Create Web Service"

### Step 4: Configure Environment Variables

Add these environment variables in Render Dashboard:

#### Required Variables
```bash
# API Keys
POLYGON_API_KEY=your_polygon_key_here
FMP_API_KEY=your_fmp_key_here
OPENAI_API_KEY=your_openai_key_here

# Redis Configuration
REDIS_HOST=your_redis_host
REDIS_PORT=12579
REDIS_PASSWORD=your_redis_password
REDIS_DB=0

# Feature Flags
TECHNIC_USE_REDIS=1
TECHNIC_USE_ML_ALPHA=true
TECHNIC_USE_RAY=0

# Optional: API Authentication
TECHNIC_API_KEY=your_secure_api_key
```

#### How to Add Variables in Render:
1. Go to your service in Render Dashboard
2. Click "Environment" tab
3. Click "Add Environment Variable"
4. Enter key and value
5. Click "Save Changes"
6. Service will automatically redeploy

### Step 5: Verify Deployment

#### Check Health Endpoint
```bash
curl https://technic-m5vn.onrender.com/health
```

Expected response:
```json
{"status": "ok"}
```

#### Check Version Endpoint
```bash
curl https://technic-m5vn.onrender.com/version
```

Expected response:
```json
{
  "api_version": "1.0.0",
  "use_ml_alpha": true,
  "use_tft_features": false
}
```

#### Test Scan Endpoint
```bash
curl -X POST https://technic-m5vn.onrender.com/v1/scan \
  -H "Content-Type: application/json" \
  -d '{
    "max_symbols": 10,
    "trade_style": "Swing",
    "min_tech_rating": 0.0,
    "options_mode": "stock_plus_options"
  }'
```

Expected: JSON response with scan results

### Step 6: Test Mobile App Connection

1. Open the mobile app in emulator
2. Navigate to Scanner screen
3. Tap "Scan" button
4. Verify:
   - ✅ No 404 errors in logs
   - ✅ Scan results appear
   - ✅ Data loads correctly
   - ✅ No crashes

---

## 🔧 Troubleshooting

### Issue: 404 Not Found

**Symptom**: Mobile app shows 404 errors
**Cause**: Wrong endpoint or API not deployed
**Solution**:
1. Verify Render service is running
2. Check logs in Render Dashboard
3. Confirm endpoint is `/v1/scan` not `/scan`
4. Test with curl command above

### Issue: 500 Internal Server Error

**Symptom**: API returns 500 errors
**Cause**: Missing environment variables or dependencies
**Solution**:
1. Check Render logs for error details
2. Verify all environment variables are set
3. Check if Redis is accessible
4. Verify API keys are valid

### Issue: Timeout Errors

**Symptom**: Requests timeout after 30 seconds
**Cause**: Scan taking too long or cold start
**Solution**:
1. Reduce `max_symbols` in request
2. Upgrade to higher Render plan
3. Enable Redis caching
4. Keep service warm with health checks

### Issue: Memory Errors

**Symptom**: Service crashes with OOM errors
**Cause**: ML models using too much memory
**Solution**:
1. Upgrade to Render Pro plan (more RAM)
2. Reduce model complexity
3. Implement model caching
4. Use Lambda for heavy computations

---

## 📊 Monitoring

### Render Dashboard
- **Logs**: View real-time logs
- **Metrics**: CPU, memory, response times
- **Events**: Deployments, restarts, errors
- **Health**: Uptime and health check status

### Key Metrics to Monitor
- **Response Time**: Should be <5 seconds for scans
- **Error Rate**: Should be <1%
- **Memory Usage**: Should stay under 80%
- **CPU Usage**: Spikes during scans are normal

### Alerts
Set up alerts in Render for:
- Service down
- High error rate
- High memory usage
- Deployment failures

---

## 🔐 Security

### API Key Authentication
The API supports optional API key authentication:

```python
# In mobile app, add header:
headers = {
    'X-API-Key': 'your_api_key_here'
}
```

To enable:
1. Set `TECHNIC_API_KEY` environment variable in Render
2. Update mobile app to include API key in requests
3. Test authentication works

### CORS Configuration
CORS is already configured in `api_server.py` to allow:
- `http://localhost:*`
- `https://technic-m5vn.onrender.com`
- `https://*.onrender.com`

Mobile apps don't need CORS, but web apps do.

---

## 📈 Performance Optimization

### Redis Caching
Enable Redis to cache:
- Universe data
- Price history
- Fundamentals
- Scan results

Benefits:
- 10x faster response times
- Reduced API costs
- Better user experience

### Cold Start Mitigation
Render free tier has cold starts. Solutions:
1. Upgrade to paid plan (no cold starts)
2. Use health check pings to keep warm
3. Implement lazy loading in mobile app
4. Show loading states to users

### Scaling
As usage grows:
1. **Horizontal**: Add more instances
2. **Vertical**: Upgrade to larger plan
3. **Caching**: Implement aggressive caching
4. **CDN**: Use CDN for static assets
5. **Database**: Move to dedicated database

---

## 🎯 Next Steps After Deployment

### 1. Test All Endpoints
- [ ] `/health` - Health check
- [ ] `/version` - Version info
- [ ] `/v1/scan` - Scanner
- [ ] `/v1/copilot` - AI assistant
- [ ] `/v1/symbol/{ticker}` - Symbol details
- [ ] `/universe_stats` - Universe data

### 2. Update Mobile App
- [ ] Verify API base URL
- [ ] Test all API calls
- [ ] Handle errors gracefully
- [ ] Add loading states
- [ ] Implement retry logic

### 3. Monitor Performance
- [ ] Set up Render alerts
- [ ] Monitor response times
- [ ] Track error rates
- [ ] Review logs regularly

### 4. Optimize
- [ ] Enable Redis caching
- [ ] Implement request batching
- [ ] Add response compression
- [ ] Optimize database queries

### 5. Documentation
- [ ] Document API endpoints
- [ ] Create API reference
- [ ] Write integration guide
- [ ] Add troubleshooting tips

---

## 📚 Resources

### Render Documentation
- [Render Docs](https://render.com/docs)
- [Blueprint Spec](https://render.com/docs/blueprint-spec)
- [Environment Variables](https://render.com/docs/environment-variables)
- [Health Checks](https://render.com/docs/health-checks)

### FastAPI Documentation
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Deployment](https://fastapi.tiangolo.com/deployment/)
- [CORS](https://fastapi.tiangolo.com/tutorial/cors/)

### Mobile App Integration
- [API Service Code](technic_mobile/lib/services/api_service.dart)
- [Testing Guide](TESTING_QUICK_REFERENCE.md)
- [Build Documentation](BUILD_SUCCESS_SUMMARY.md)

---

## ✅ Deployment Checklist

### Pre-Deployment
- [x] Create `render.yaml` configuration
- [x] Verify `requirements.txt` is complete
- [x] Test API locally
- [x] Document environment variables
- [ ] Commit changes to Git
- [ ] Push to GitHub

### Deployment
- [ ] Create Render service
- [ ] Configure environment variables
- [ ] Deploy service
- [ ] Verify health endpoint
- [ ] Test scan endpoint

### Post-Deployment
- [ ] Test mobile app connection
- [ ] Monitor logs for errors
- [ ] Set up alerts
- [ ] Document API endpoints
- [ ] Update mobile app if needed

### Verification
- [ ] No 404 errors in mobile app
- [ ] Scan results load correctly
- [ ] All endpoints responding
- [ ] Performance acceptable
- [ ] No crashes or errors

---

## 🎉 Success Criteria

### Deployment Successful When:
1. ✅ Render service is running
2. ✅ Health endpoint returns 200 OK
3. ✅ Scan endpoint returns results
4. ✅ Mobile app connects successfully
5. ✅ No errors in logs
6. ✅ Response times acceptable
7. ✅ All features working

---

**Status**: 📝 Ready for Deployment
**Next Action**: Commit changes and push to GitHub
**Estimated Time**: 15-30 minutes

---

*Last Updated: December 19, 2024*
*Deployment Guide v1.0*
