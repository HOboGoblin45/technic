# Backend Fix Required for Render Deployment

## Issue

The Render API is returning 500 errors when the Flutter app calls `/v1/scan`:

```
AttributeError: 'ScanRequest' object has no attribute 'options_mode'
```

## Root Cause

The `ScanRequest` Pydantic model in `technic_v4/api_server.py` (line ~187) is missing the `options_mode` field, but the code tries to access it:

```python
options_mode = req.options_mode or "stock_plus_options"  # Line 187
```

## Solution

Update `technic_v4/api_server.py`:

### 1. Add `options_mode` to ScanRequest model:

```python
class ScanRequest(BaseModel):
    universe: Optional[List[str]] = None
    max_symbols: int = 25
    trade_style: str = "Short-term swing"
    min_tech_rating: float = 0.0
    options_mode: Optional[str] = "stock_plus_options"  # ADD THIS LINE
```

### 2. Redeploy to Render

After making this change, redeploy the service to Render.

## Workaround (Temporary)

Until the backend is fixed, the Flutter app will show errors when trying to scan. The app itself is working correctly - this is purely a backend API issue.

## Flutter App Status

✅ All 9 user issues are fixed in the Flutter app
✅ API authentication is configured correctly  
✅ Request format is correct
⚠️ Backend needs update to accept requests

Once the backend is updated and redeployed, the scanner will work perfectly.
