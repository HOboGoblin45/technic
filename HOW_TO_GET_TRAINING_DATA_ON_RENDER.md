# How to Get training_data.parquet on Render (Without Git LFS)

## The Problem
GitHub LFS budget exceeded, but you want the training_data.parquet file on Render for the meta_experience feature.

---

## ✅ SOLUTION 1: Upload to Cloud Storage (RECOMMENDED)

### Best Approach: Use a Free Cloud Storage Service

**Option A: Google Drive (Free)**

1. **Upload file to Google Drive:**
   - Upload `data/training_data.parquet` to your Google Drive
   - Right-click → Share → Get link
   - Change to "Anyone with the link can view"
   - Copy the file ID from the URL

2. **Download in Render build:**

Create `download_training_data.sh`:
```bash
#!/bin/bash
# Download training data from Google Drive

FILE_ID="YOUR_GOOGLE_DRIVE_FILE_ID"
DEST="data/training_data.parquet"

# Create data directory
mkdir -p data

# Download using gdown (install in requirements.txt)
pip install gdown
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${DEST}"

echo "Training data downloaded successfully"
```

3. **Add to requirements.txt:**
```
gdown>=4.7.1
```

4. **Update Render build command:**
```bash
chmod +x download_training_data.sh && ./download_training_data.sh && python -m uvicorn api:app --host 0.0.0.0 --port $PORT
```

**Option B: AWS S3 (Free Tier)**

1. Upload to S3 bucket (make public or use presigned URL)
2. Download in build:
```bash
curl -o data/training_data.parquet "https://your-bucket.s3.amazonaws.com/training_data.parquet"
```

**Option C: Dropbox (Free)**

1. Upload to Dropbox
2. Get public link
3. Download in build:
```bash
wget -O data/training_data.parquet "https://www.dropbox.com/s/YOUR_LINK/training_data.parquet?dl=1"
```

---

## ✅ SOLUTION 2: Commit File Directly to Git (Simple)

### If File is Small Enough (<100MB)

1. **Remove from LFS tracking:**
```bash
git lfs untrack "data/training_data.parquet"
git add .gitattributes
git commit -m "Remove training_data.parquet from LFS"
```

2. **Add file normally:**
```bash
git rm --cached data/training_data.parquet
git add data/training_data.parquet
git commit -m "Add training_data.parquet as regular file"
git push origin main
```

**Note:** GitHub has a 100MB file size limit. If your file is larger, use Solution 1 or 3.

---

## ✅ SOLUTION 3: Generate File on Render (Best Long-Term)

### Rebuild training data from scan history

The code already has a fallback to use:
```
technic_v4/scanner_output/history/replay_ics.parquet
```

**Option A: Pre-generate on first deploy**

Add to Render build script:
```bash
# Generate training data from historical scans
python scripts/build_training_data.py --output data/training_data.parquet

# Or use existing scan history
cp technic_v4/scanner_output/history/replay_ics.parquet data/training_data.parquet
```

**Option B: Let it build naturally**

The meta_experience module will automatically use `replay_ics.parquet` as a fallback. After a few scans, you'll have your own production training data!

---

## ✅ SOLUTION 4: Use Render Persistent Disk

### Store file on Render's persistent storage

1. **In Render Dashboard:**
   - Add a Persistent Disk (free tier: 1GB)
   - Mount at `/opt/render/project/data`

2. **Upload file once via Render Shell:**
```bash
# SSH into Render
render shell

# Upload file (use scp or curl)
curl -o /opt/render/project/data/training_data.parquet "YOUR_CLOUD_URL"
```

3. **File persists across deployments!**

---

## 🎯 RECOMMENDED APPROACH

**For immediate deployment:**
1. Set `GIT_LFS_SKIP_SMUDGE=1` (deploy works immediately)
2. Use **Solution 1 (Google Drive)** to download file during build
3. File will be available, feature works!

**For long-term:**
- Use **Solution 3** - generate from your own scan history
- This gives you real production data instead of old training data
- More accurate and up-to-date!

---

## IMPLEMENTATION STEPS (Google Drive Method)

### Step 1: Upload to Google Drive
1. Go to drive.google.com
2. Upload `data/training_data.parquet`
3. Right-click → Share → Copy link
4. Extract file ID from URL:
   ```
   https://drive.google.com/file/d/1ABC123XYZ/view
                                  ^^^^^^^^^ This is the file ID
   ```

### Step 2: Create Download Script

Create `download_training_data.sh`:
```bash
#!/bin/bash
FILE_ID="YOUR_FILE_ID_HERE"
mkdir -p data
pip install -q gdown
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O data/training_data.parquet
echo "✅ Training data downloaded"
```

### Step 3: Update Render Build Command

In Render dashboard, set build command to:
```bash
chmod +x download_training_data.sh && ./download_training_data.sh && pip install -r requirements.txt
```

### Step 4: Add gdown to requirements.txt
```
gdown>=4.7.1
```

### Step 5: Deploy!

---

## RESULT

✅ Render deploys successfully  
✅ Training data downloaded during build  
✅ Meta experience feature works  
✅ No features lost  
✅ No Git LFS issues  

**You get everything working without Git LFS!**
