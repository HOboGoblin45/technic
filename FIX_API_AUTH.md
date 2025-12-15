# Fix API Authentication Issue

## 🔍 Problem

The API is returning **401 Unauthorized** because it's checking for an API key, but you haven't set one yet.

## ✅ Solution: Disable API Key (Dev Mode)

The API has a dev mode that allows all requests when `TECHNIC_API_KEY` is not set.

### **Option 1: Remove API Key from Render (Recommended for Testing)**

1. Go to your Render dashboard: https://dashboard.render.com
2. Click on your "technic" service
3. Go to "Environment" tab
4. Look for `TECHNIC_API_KEY` variable
5. **Delete it** (or set it to empty)
6. Click "Save Changes"
7. Render will redeploy automatically

**After this, all endpoints will work without authentication!**

---

### **Option 2: Use API Key (Production Mode)**

If you want to keep authentication:

#### **1. Get Your API Key from Render**

1. Go to Render dashboard
2. Click on "technic" service
3. Go to "Environment" tab
4. Find `TECHNIC_API_KEY` value
5. Copy it

#### **2. Use API Key in PowerShell**

```powershell
# Set your API key
$apiKey = "your-api-key-here"

# Health check (no auth needed)
Invoke-RestMethod https://technic-m5vn.onrender.com/health

# Scan with API key
$headers = @{
    "X-API-Key" = $apiKey
    "Content-Type" = "application/json"
}

$body = @{
    max_symbols = 10
    min_tech_rating = 0.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "https://technic-m5vn.onrender.com/v1/scan" `
    -Method Post `
    -Headers $headers `
    -Body $body
```

#### **3. Symbol Detail with API Key**

```powershell
$headers = @{
    "X-API-Key" = $apiKey
}

Invoke-RestMethod -Uri "https://technic-m5vn.onrender.com/v1/symbol/AAPL" `
    -Headers $headers
```

---

## 🎯 Recommended: Disable Auth for Testing

For now, I recommend **Option 1** (remove the API key) so you can test freely.

### **Steps:**

1. **Go to Render Dashboard**
   ```
   https://dashboard.render.com
   ```

2. **Click on "technic" service**

3. **Go to "Environment" tab**

4. **Find `TECHNIC_API_KEY`**

5. **Delete it or set to empty**

6. **Save Changes**

7. **Wait for redeploy** (~30-60 seconds with your optimized Docker!)

8. **Test again:**
   ```powershell
   Invoke-RestMethod -Uri "https://technic-m5vn.onrender.com/v1/scan" `
       -Method Post `
       -Body '{"max_symbols":10,"min_tech_rating":0.0}' `
       -ContentType "application/json"
   ```

---

## 📊 How API Auth Works

From `api_server.py`:

```python
def get_api_key(x_api_key: Optional[str] = Header(default=None)) -> str:
    expected = os.getenv("TECHNIC_API_KEY")
    if expected is None:
        return ""  # Dev mode - allow all!
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key
```

**Key Points:**
- If `TECHNIC_API_KEY` env var is **not set** → All requests allowed ✅
- If `TECHNIC_API_KEY` env var **is set** → Must provide `X-API-Key` header ❌

---

## 🔧 Quick Fix Commands

### **After Removing API Key from Render:**

```powershell
# Test scan
Invoke-RestMethod -Uri "https://technic-m5vn.onrender.com/v1/scan" `
    -Method Post `
    -Body '{"max_symbols":10,"min_tech_rating":0.0}' `
    -ContentType "application/json"

# Test symbol
Invoke-RestMethod https://technic-m5vn.onrender.com/v1/symbol/AAPL
```

---

## ✅ Summary

**Problem:** API requires authentication
**Solution:** Remove `TECHNIC_API_KEY` from Render environment variables
**Result:** All endpoints work without auth (perfect for testing!)

Once you remove the API key, you'll be able to test all endpoints freely!
