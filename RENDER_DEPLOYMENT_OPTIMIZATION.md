# Render Deployment Speed Optimization Guide

## Current Status
- **Current build time:** ~15 minutes
- **Main bottleneck:** Installing PyTorch (900MB), CUDA libraries (2GB+), and other ML dependencies

## Quick Wins (Implement Now)

### 1. Enable Docker Layer Caching ✅ HIGHEST IMPACT
**Savings: 10-12 minutes on subsequent builds**

Render automatically caches Docker layers, but we can optimize the Dockerfile to maximize cache hits:

**Current Dockerfile Issue:**
- Every code change invalidates the pip install cache
- All 3GB+ of dependencies reinstall every time

**Solution: Reorder Dockerfile layers**

```dockerfile
FROM python:3.10.11-slim

WORKDIR /app

# System dependencies (cached unless apt packages change)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY requirements first (cached unless requirements.txt changes)
COPY requirements.txt ./

# Install dependencies (CACHED unless requirements.txt changes!)
RUN pip install --no-cache-dir -r requirements.txt

# Copy code LAST (changes frequently, but doesn't invalidate pip cache)
COPY technic_v4 ./technic_v4
COPY models ./models
COPY start.sh ./start.sh
RUN chmod +x start.sh

ENV PYTHONUNBUFFERED=1
EXPOSE 8502

CMD ["bash", "start.sh"]
```

**Impact:**
- ✅ First build: 15 minutes (same)
- ✅ Code changes: 30-60 seconds (only copies new code!)
- ✅ Dependency changes: 15 minutes (only when requirements.txt changes)

---

### 2. Use Pre-built Docker Image (Advanced)
**Savings: 12-14 minutes on ALL builds**

Instead of building from scratch, use a pre-built image with all dependencies:

**Option A: Use Official PyTorch Image**
```dockerfile
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /app

# Only install non-PyTorch dependencies
COPY requirements-minimal.txt ./
RUN pip install --no-cache-dir -r requirements-minimal.txt

COPY technic_v4 ./technic_v4
COPY models ./models
COPY start.sh ./start.sh
RUN chmod +x start.sh

CMD ["bash", "start.sh"]
```

**Create requirements-minimal.txt:**
```
# Remove torch, torchvision from requirements.txt
# Keep only: streamlit, fastapi, pandas, etc.
```

**Impact:**
- ✅ All builds: 2-3 minutes (PyTorch pre-installed!)

---

### 3. Reduce Dependency Size
**Savings: 3-5 minutes**

**Current Issue:** Installing packages we might not need

**Review and remove unused dependencies:**
```bash
# Do you actually use these in production?
pytorch-forecasting  # 391 KB + dependencies
shap                 # 1.0 MB + heavy deps
pytorch-lightning    # 849 KB + dependencies
```

**If not needed in production, create two requirements files:**

**requirements-dev.txt** (for local development):
```
# All dependencies including ML training tools
pytorch-forecasting>=1.0.0
shap>=0.49
pytorch-lightning>=2.6
```

**requirements.txt** (for production):
```
# Only runtime dependencies
torch>=2.9
fastapi>=0.110
streamlit>=1.28
# ... etc
```

---

### 4. Use Render's Build Cache (Already Enabled)
**Status:** ✅ Already working

Render caches:
- Docker layers
- pip packages
- Build artifacts

**To verify cache is working:**
- Look for "Downloaded 4.5GB in 11s" in logs (cache hit!)
- vs "Downloading..." (cache miss)

---

## Medium-Term Optimizations

### 5. Split Backend and Frontend Services
**Savings: Deploy only what changed**

**Current:** One monolithic service (backend + frontend)
**Better:** Separate services

```
technic-backend/     # FastAPI only, deploys rarely
technic-frontend/    # Streamlit only, deploys frequently
```

**Impact:**
- Frontend changes: 1-2 minute deploys
- Backend changes: 15 minute deploys (but rare)

---

### 6. Use Render's Native Buildpacks (Instead of Docker)
**Savings: 5-7 minutes**

Render's native Python buildpack is faster than Docker for simple apps:

**In Render Dashboard:**
- Build Command: `pip install -r requirements.txt`
- Start Command: `bash start.sh`
- Environment: Python 3.10

**Pros:**
- Faster builds
- Automatic caching
- Less configuration

**Cons:**
- Less control
- May not work with complex setups

---

## Long-Term Optimizations

### 7. Use Render's Persistent Disk for Dependencies
**Savings: 10-12 minutes**

Install dependencies to persistent disk once, reuse forever:

**start.sh:**
```bash
#!/bin/bash

# Check if dependencies are already installed
if [ ! -d "/opt/render/project/.venv" ]; then
    echo "Installing dependencies (first time only)..."
    python -m venv /opt/render/project/.venv
    source /opt/render/project/.venv/bin/activate
    pip install -r requirements.txt
else
    echo "Using cached dependencies..."
    source /opt/render/project/.venv/bin/activate
fi

# Create symlink for training data
ln -sf /opt/render/project/data/training_data_v2.parquet data/training_data_v2.parquet

# Start server
uvicorn technic_v4.api_server:app --host 0.0.0.0 --port ${PORT:-10000}
```

**Impact:**
- First deploy: 15 minutes
- All subsequent deploys: 2-3 minutes!

---

### 8. Use GitHub Actions for Pre-built Images
**Savings: 12-14 minutes**

Build Docker image in GitHub Actions, push to Docker Hub, Render pulls pre-built image:

**Workflow:**
1. Push code to GitHub
2. GitHub Actions builds Docker image (parallel, faster)
3. Push image to Docker Hub
4. Render pulls pre-built image (30 seconds!)

**Impact:**
- Render deploy: 30-60 seconds
- Total time: 5-7 minutes (GitHub Actions build time)

---

## Recommended Implementation Order

### Phase 1: Immediate (Do Now) ⚡
1. **Reorder Dockerfile layers** (5 minutes to implement)
   - Savings: 10-12 minutes per deploy
   - Effort: Very low
   - **DO THIS FIRST!**

### Phase 2: This Week 📅
2. **Review and remove unused dependencies** (30 minutes)
   - Savings: 3-5 minutes per deploy
   - Effort: Low

3. **Use persistent disk for dependencies** (1 hour)
   - Savings: 10-12 minutes per deploy
   - Effort: Medium

### Phase 3: Next Sprint 🚀
4. **Split backend/frontend services** (2-4 hours)
   - Savings: Deploy only what changed
   - Effort: Medium-High

5. **Use pre-built Docker image** (4-6 hours)
   - Savings: 12-14 minutes per deploy
   - Effort: High

---

## Expected Results

### Current State
- Every deploy: 15 minutes
- Code change: 15 minutes
- Dependency change: 15 minutes

### After Phase 1 (Reorder Dockerfile)
- First deploy: 15 minutes
- Code change: **30-60 seconds** ⚡
- Dependency change: 15 minutes

### After Phase 2 (Persistent Disk)
- First deploy: 15 minutes
- Code change: **30-60 seconds** ⚡
- Dependency change: **2-3 minutes** ⚡

### After Phase 3 (Pre-built Image)
- All deploys: **30-60 seconds** ⚡⚡⚡

---

## Quick Start: Implement Phase 1 Now

I can immediately optimize your Dockerfile to enable proper layer caching. This will reduce most deploys from 15 minutes to under 1 minute.

**Would you like me to:**
1. ✅ Reorder Dockerfile layers now (5 min, huge impact)
2. ✅ Set up persistent disk for dependencies (30 min, huge impact)
3. ⏸️ Review unused dependencies (can do later)
4. ⏸️ Advanced optimizations (can do later)

**Recommendation:** Do #1 immediately, then #2 this week!
