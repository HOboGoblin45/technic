# Fix Render Deployment - api_hybrid Import Error

## Problem
```
ERROR: Error loading ASGI app. Could not import module "api_hybrid".
```

## Root Cause
Render cannot find or import the `api_hybrid.py` module during deployment.

## Solutions

### Solution 1: Use api.py Instead (Recommended)
The simplest fix is to use the working `api.py` file instead of `api_hybrid.py`.

**Update start.sh:**
```bash
#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Create symlink to training data on persistent disk
if [ -f "/opt/render/project/data/training_data_v2.parquet" ]; then
    ln -sf /opt/render/project/data/training_data_v2.parquet data/training_data_v2.parquet
    echo "✅ Symlink created for training_data_v2.parquet"
else
    echo "⚠️  Warning: training_data_v2.parquet not found on persistent disk"
fi

# Start the API server (use api.py which is proven to work)
exec python -m uvicorn api:app --host 0.0.0.0 --port "$PORT"
```

### Solution 2: Verify api_hybrid.py is Committed
Check if `api_hybrid.py` is in your git repository:

```bash
git status
git add api_hybrid.py
git commit -m "Add api_hybrid.py"
git push
```

### Solution 3: Check .gitignore
Make sure `api_hybrid.py` is not being ignored:

```bash
# Check if it's ignored
git check-ignore api_hybrid.py

# If it returns the file name, remove it from .gitignore
```

### Solution 4: Fallback Logic in start.sh
Add fallback logic to use `api.py` if `api_hybrid.py` fails:

```bash
#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Create symlink to training data on persistent disk
if [ -f "/opt/render/project/data/training_data_v2.parquet" ]; then
    ln -sf /opt/render/project/data/training_data_v2.parquet data/training_data_v2.parquet
    echo "✅ Symlink created for training_data_v2.parquet"
else
    echo "⚠️  Warning: training_data_v2.parquet not found on persistent disk"
fi

# Try to start with api_hybrid, fallback to api.py
if python -c "import api_hybrid" 2>/dev/null; then
    echo "✅ Starting with api_hybrid.py"
    exec python -m uvicorn api_hybrid:app --host 0.0.0.0 --port "$PORT"
else
    echo "⚠️  api_hybrid.py not found, falling back to api.py"
    exec python -m uvicorn api:app --host 0.0.0.0 --port "$PORT"
fi
```

## Quick Fix (Immediate)

**Run this command to update start.sh:**

```bash
cat > start.sh << 'EOF'
#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Create symlink to training data on persistent disk
if [ -f "/opt/render/project/data/training_data_v2.parquet" ]; then
    ln -sf /opt/render/project/data/training_data_v2.parquet data/training_data_v2.parquet
    echo "✅ Symlink created for training_data_v2.parquet"
else
    echo "⚠️  Warning: training_data_v2.parquet not found on persistent disk"
fi

# Use api.py (proven to work)
exec python -m uvicorn api:app --host 0.0.0.0 --port "$PORT"
EOF

chmod +x start.sh
git add start.sh
git commit -m "Fix: Use api.py instead of api_hybrid.py for Render deployment"
git push
```

## Verification

After pushing the fix, check Render logs for:
```
✅ Symlink created for training_data_v2.parquet
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:PORT
```

## Why This Happened

The `api_hybrid.py` file may:
1. Not be committed to git
2. Be in .gitignore
3. Have import errors on Render's environment
4. Be missing dependencies

Using `api.py` is safer as it's the proven, working version.

## Next Steps

1. **Apply Solution 1** (use api.py) - Safest and quickest
2. **Push to git**
3. **Wait for Render to redeploy** (automatic)
4. **Verify deployment** succeeds
5. **Test API** endpoints

---

**Recommended Action**: Use Solution 1 (switch to api.py)
**Estimated Fix Time**: 2-3 minutes
**Risk**: Low (api.py is proven to work)
