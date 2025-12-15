# Technic API Testing - Windows PowerShell Commands

## 🪟 Windows PowerShell Specific Commands

PowerShell has different syntax than bash. Use these commands:

---

## ✅ Health Check

```powershell
curl https://technic-m5vn.onrender.com/health
```

---

## 📊 Test Scanner - Small Scan (10 symbols)

```powershell
$body = @{
    max_symbols = 10
    min_tech_rating = 0.0
    universe_name = "us_core"
} | ConvertTo-Json

Invoke-RestMethod -Uri "https://technic-m5vn.onrender.com/v1/scan" -Method Post -Body $body -ContentType "application/json"
```

---

## 🚀 Test Scanner - Full Scan (5000 symbols)

```powershell
$body = @{
    max_symbols = 5000
    min_tech_rating = 0.0
    universe_name = "us_core"
} | ConvertTo-Json

Measure-Command {
    Invoke-RestMethod -Uri "https://technic-m5vn.onrender.com/v1/scan" -Method Post -Body $body -ContentType "application/json"
}
```

---

## 📈 Get Symbol Details

```powershell
Invoke-RestMethod -Uri "https://technic-m5vn.onrender.com/v1/symbol/AAPL" -Method Get
```

---

## 🎯 Alternative: Use Invoke-WebRequest for More Details

```powershell
# Small scan with full response details
$response = Invoke-WebRequest -Uri "https://technic-m5vn.onrender.com/v1/scan" `
    -Method Post `
    -ContentType "application/json" `
    -Body '{"max_symbols": 10, "min_tech_rating": 0.0, "universe_name": "us_core"}'

# View status code
$response.StatusCode

# View content
$response.Content | ConvertFrom-Json
```

---

## 🔧 Quick Test Script

Save this as `test-technic.ps1`:

```powershell
Write-Host "🔍 Testing Technic API..." -ForegroundColor Cyan
Write-Host ""

# 1. Health Check
Write-Host "1. Health Check..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "https://technic-m5vn.onrender.com/health"
    Write-Host "✅ Status: $($health.status)" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed: $_" -ForegroundColor Red
}
Write-Host ""

# 2. Small Scan
Write-Host "2. Small Scan (10 symbols)..." -ForegroundColor Yellow
try {
    $body = @{
        max_symbols = 10
        min_tech_rating = 0.0
        universe_name = "us_core"
    } | ConvertTo-Json
    
    $scan = Invoke-RestMethod -Uri "https://technic-m5vn.onrender.com/v1/scan" -Method Post -Body $body -ContentType "application/json"
    Write-Host "✅ Found $($scan.results.Count) results" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed: $_" -ForegroundColor Red
}
Write-Host ""

# 3. Symbol Detail
Write-Host "3. Symbol Detail (AAPL)..." -ForegroundColor Yellow
try {
    $symbol = Invoke-RestMethod -Uri "https://technic-m5vn.onrender.com/v1/symbol/AAPL"
    Write-Host "✅ MERIT Score: $($symbol.merit_score)" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed: $_" -ForegroundColor Red
}
Write-Host ""

Write-Host "✅ All tests complete!" -ForegroundColor Green
```

Run with:
```powershell
.\test-technic.ps1
```

---

## 🌐 Or Use Your Browser!

The easiest way to test on Windows:

1. **Open Interactive Docs:**
   ```
   https://technic-m5vn.onrender.com/docs
   ```

2. **Click on `/v1/scan` endpoint**

3. **Click "Try it out"**

4. **Enter parameters:**
   ```json
   {
     "max_symbols": 10,
     "min_tech_rating": 0.0,
     "universe_name": "us_core"
   }
   ```

5. **Click "Execute"**

---

## 💡 Why PowerShell is Different

PowerShell uses:
- `Invoke-RestMethod` instead of `curl -X POST`
- `@{}` hashtables for JSON bodies
- `ConvertTo-Json` to convert to JSON
- Different quote escaping rules

---

## 🎯 Quick Copy-Paste Commands

### Test 1: Health Check
```powershell
Invoke-RestMethod https://technic-m5vn.onrender.com/health
```

### Test 2: Small Scan
```powershell
Invoke-RestMethod -Uri "https://technic-m5vn.onrender.com/v1/scan" -Method Post -Body '{"max_symbols":10,"min_tech_rating":0.0,"universe_name":"us_core"}' -ContentType "application/json"
```

### Test 3: Symbol Details
```powershell
Invoke-RestMethod https://technic-m5vn.onrender.com/v1/symbol/AAPL
```

---

## ✅ Expected Results

### Health Check:
```json
{
  "status": "ok"
}
```

### Small Scan:
```json
{
  "results": [
    {
      "symbol": "AAPL",
      "merit_score": 8.5,
      "technical_rating": 7.8,
      ...
    }
  ],
  "scan_time": 5.2,
  "symbols_scanned": 10
}
```

### Symbol Detail:
```json
{
  "symbol": "AAPL",
  "merit_score": 8.5,
  "price": 175.50,
  "indicators": {...}
}
```

---

## 🚀 Your API is Live!

**Base URL:** https://technic-m5vn.onrender.com

**Interactive Docs:** https://technic-m5vn.onrender.com/docs

Use the PowerShell commands above or the browser-based docs for easy testing!
