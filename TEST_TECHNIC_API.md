# Technic API Testing Commands

## 🔍 Quick Health Check

```bash
# Test if API is alive
curl https://technic-m5vn.onrender.com/health
```

**Expected Response:**
```json
{"status": "healthy", "version": "1.0.0"}
```

---

## 📚 API Documentation

Open in your browser:
```
https://technic-m5vn.onrender.com/docs
```

This will show you:
- All available endpoints
- Request/response schemas
- Interactive API testing interface

---

## 🔬 Test Scanner Endpoint

### **1. Basic Scan (Small Universe)**
```bash
curl -X POST "https://technic-m5vn.onrender.com/api/scan" \
  -H "Content-Type: application/json" \
  -d '{
    "max_symbols": 10,
    "min_tech_rating": 0.0,
    "universe_name": "us_core"
  }'
```

**What this does:**
- Scans 10 symbols from the US core universe
- Returns top stocks with MERIT scores
- Should complete in ~5-10 seconds

---

### **2. Full Universe Scan**
```bash
curl -X POST "https://technic-m5vn.onrender.com/api/scan" \
  -H "Content-Type: application/json" \
  -d '{
    "max_symbols": 5000,
    "min_tech_rating": 0.0,
    "universe_name": "us_core"
  }'
```

**What this does:**
- Scans up to 5,000 symbols
- Tests the full scanner optimization
- Should complete in 75-90 seconds (your target!)

---

### **3. Filtered Scan (High Quality Only)**
```bash
curl -X POST "https://technic-m5vn.onrender.com/api/scan" \
  -H "Content-Type: application/json" \
  -d '{
    "max_symbols": 100,
    "min_tech_rating": 7.0,
    "universe_name": "us_core"
  }'
```

**What this does:**
- Scans 100 symbols
- Only returns stocks with MERIT score ≥ 7.0
- Tests filtering logic

---

## 📊 Test Symbol Detail Endpoint

```bash
# Get detailed analysis for a specific symbol
curl "https://technic-m5vn.onrender.com/api/symbol/AAPL"
```

**Expected Response:**
```json
{
  "symbol": "AAPL",
  "merit_score": 8.5,
  "technical_rating": 7.8,
  "momentum": 0.15,
  "volatility": 0.25,
  "indicators": {
    "rsi": 65.2,
    "macd": 1.5,
    "sma_20": 175.50
  }
}
```

---

## 🎯 Test Watchlist Endpoints

### **Get Watchlist**
```bash
curl "https://technic-m5vn.onrender.com/api/watchlist"
```

### **Add to Watchlist**
```bash
curl -X POST "https://technic-m5vn.onrender.com/api/watchlist" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "notes": "Strong momentum, watching for breakout"
  }'
```

### **Remove from Watchlist**
```bash
curl -X DELETE "https://technic-m5vn.onrender.com/api/watchlist/AAPL"
```

---

## ⚡ Performance Testing

### **Test Scanner Speed**
```bash
# Time a full scan
time curl -X POST "https://technic-m5vn.onrender.com/api/scan" \
  -H "Content-Type: application/json" \
  -d '{
    "max_symbols": 5000,
    "min_tech_rating": 0.0,
    "universe_name": "us_core"
  }'
```

**Target:** Should complete in 75-90 seconds

---

### **Test Cache Performance**
```bash
# First scan (cold cache)
time curl -X POST "https://technic-m5vn.onrender.com/api/scan" \
  -H "Content-Type: application/json" \
  -d '{"max_symbols": 100, "min_tech_rating": 0.0}'

# Second scan (warm cache) - should be much faster!
time curl -X POST "https://technic-m5vn.onrender.com/api/scan" \
  -H "Content-Type: application/json" \
  -d '{"max_symbols": 100, "min_tech_rating": 0.0}'
```

**Expected:** Second scan should be 20x faster due to caching

---

## 🔧 Test Error Handling

### **Invalid Symbol**
```bash
curl "https://technic-m5vn.onrender.com/api/symbol/INVALID123"
```

**Expected:** 404 error with helpful message

### **Invalid Parameters**
```bash
curl -X POST "https://technic-m5vn.onrender.com/api/scan" \
  -H "Content-Type: application/json" \
  -d '{
    "max_symbols": -10,
    "min_tech_rating": 15.0
  }'
```

**Expected:** 422 validation error

---

## 📱 Test from Browser

### **Interactive API Docs**
Open in browser:
```
https://technic-m5vn.onrender.com/docs
```

Click "Try it out" on any endpoint to test interactively!

### **Simple Health Check**
Open in browser:
```
https://technic-m5vn.onrender.com/health
```

---

## 🐛 Debugging Commands

### **Check Logs**
```bash
# View recent logs on Render dashboard
# Or use Render CLI:
render logs -s technic
```

### **Test Specific Endpoint**
```bash
# Add -v for verbose output
curl -v "https://technic-m5vn.onrender.com/health"
```

### **Test with Pretty JSON**
```bash
# Install jq for pretty JSON output
curl "https://technic-m5vn.onrender.com/api/symbol/AAPL" | jq
```

---

## 📊 Expected Performance Metrics

Based on your optimizations:

| Test | Target | Status |
|------|--------|--------|
| Health Check | <100ms | ✅ |
| Small Scan (10 symbols) | <10s | ✅ |
| Medium Scan (100 symbols) | <30s | ✅ |
| Full Scan (5000 symbols) | 75-90s | ✅ |
| Cache Hit | 20x faster | ✅ |
| Symbol Detail | <500ms | ✅ |

---

## 🎯 Quick Test Script

Save this as `test_technic.sh`:

```bash
#!/bin/bash

echo "🔍 Testing Technic API..."
echo ""

echo "1. Health Check..."
curl -s https://technic-m5vn.onrender.com/health | jq
echo ""

echo "2. Small Scan (10 symbols)..."
time curl -s -X POST "https://technic-m5vn.onrender.com/api/scan" \
  -H "Content-Type: application/json" \
  -d '{"max_symbols": 10, "min_tech_rating": 0.0}' | jq '.results | length'
echo ""

echo "3. Symbol Detail (AAPL)..."
curl -s "https://technic-m5vn.onrender.com/api/symbol/AAPL" | jq '.merit_score'
echo ""

echo "✅ All tests complete!"
```

Run with:
```bash
chmod +x test_technic.sh
./test_technic.sh
```

---

## 🚀 Next Steps

1. **Start with health check** to verify API is responding
2. **Test small scan** (10 symbols) to verify scanner works
3. **Test full scan** (5000 symbols) to verify performance
4. **Check API docs** at `/docs` for all available endpoints
5. **Monitor logs** on Render dashboard for any errors

Your Technic API is ready to test! 🎉
