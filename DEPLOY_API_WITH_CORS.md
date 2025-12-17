# Deploy API with CORS to Render

## ✅ What Was Changed

Added CORS middleware to `api.py` to allow Flutter web app to connect:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:*",
        "http://127.0.0.1:*",
        "https://technic-m5vn.onrender.com",
        "https://*.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🚀 Deploy to Render

### Option 1: Git Push (Recommended)

```bash
# Commit the changes
git add api.py
git commit -m "Add CORS headers for Flutter web app"
git push origin main
```

Render will automatically detect the changes and redeploy.

### Option 2: Manual Deploy

1. Go to https://dashboard.render.com
2. Find your `technic-m5vn` service
3. Click "Manual Deploy" → "Deploy latest commit"

## ⏱️ Wait for Deployment

- Deployment typically takes 2-5 minutes
- Watch the deploy logs in Render dashboard
- Wait for "Live" status

## ✅ Test the API

Once deployed, test the CORS headers:

```bash
# Test from command line
curl -H "Origin: http://localhost:8080" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: Content-Type" \
     -X OPTIONS \
     https://technic-m5vn.onrender.com/v1/scan -v
```

You should see CORS headers in the response:
```
Access-Control-Allow-Origin: http://localhost:8080
Access-Control-Allow-Methods: *
Access-Control-Allow-Headers: *
```

## 🎯 Test in Flutter App

After deployment completes:

1. **In your Flutter terminal, press `r`** to hot reload
2. **Click "Run Scan"** in the app
3. **Watch for results** instead of CORS error

### Expected Behavior

**Before (CORS Error):**
```
API error: ClientException: Failed to fetch
```

**After (Success):**
```
[API] Scan started...
[API] Received X results
```

## 🐛 If It Still Doesn't Work

### Check 1: Verify Deployment
```bash
# Check if API is live
curl https://technic-m5vn.onrender.com/health
```

Should return: `{"status":"ok"}`

### Check 2: Check CORS Headers
```bash
# Test CORS from localhost
curl -H "Origin: http://localhost:8080" \
     https://technic-m5vn.onrender.com/health -v
```

Should include: `Access-Control-Allow-Origin: http://localhost:8080`

### Check 3: Flutter App URL
Make sure your Flutter app is using the correct API URL:
- Should be: `https://technic-m5vn.onrender.com/v1/scan`
- NOT: `http://` (must be `https://`)

### Check 4: Clear Browser Cache
```
In Chrome:
1. Press F12 (open DevTools)
2. Right-click the refresh button
3. Select "Empty Cache and Hard Reload"
```

## 📊 What This Fixes

### Before CORS:
- ❌ Flutter web app couldn't connect to API
- ❌ Browser blocked cross-origin requests
- ❌ Scanner, Copilot, and all features failed

### After CORS:
- ✅ Flutter web app can connect to API
- ✅ Browser allows cross-origin requests
- ✅ Scanner, Copilot, and all features work

## 🎉 Success Indicators

You'll know it worked when:

1. **No CORS errors** in browser console
2. **Scanner returns results** when you click "Run Scan"
3. **Market movers appear** in the Ideas tab
4. **Copilot responds** to questions
5. **Symbol details load** when you tap a result

## ⏭️ Next Steps After Deployment

Once the API is deployed and working:

1. ✅ Test scanner with different sectors
2. ✅ Test Copilot with stock questions
3. ✅ Test watchlist functionality
4. ✅ Test theme toggle (dark/light)
5. ✅ Test all 5 tabs (Scanner, Ideas, Copilot, Watchlist, Settings)

---

**Status:** API updated with CORS headers. Ready to deploy to Render!

**Timeline:**
- Commit & push: 1 minute
- Render deployment: 2-5 minutes
- Testing: 2 minutes
- **Total: ~5-10 minutes**
