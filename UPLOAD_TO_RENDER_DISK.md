# Upload training_data.parquet to Render Persistent Disk

## You Have a 5GB Disk Mounted at `/opt/render/project/data`

Perfect! This is the ideal solution. Here's how to upload your file:

---

## METHOD 1: Upload via Render Shell (EASIEST)

### Step 1: Get the File URL

**Option A: Upload to Google Drive (Quick)**
1. Upload `data/training_data.parquet` to Google Drive
2. Right-click → Share → "Anyone with link can view"
3. Copy the link
4. Extract file ID from URL:
   ```
   https://drive.google.com/file/d/1ABC123XYZ/view
                                  ^^^^^^^^^ File ID
   ```

**Option B: Upload to Dropbox**
1. Upload file to Dropbox
2. Get shareable link
3. Add `?dl=1` to end of URL for direct download

### Step 2: SSH into Render

In Render Dashboard:
1. Go to your service
2. Click "Shell" tab (top right)
3. This opens a terminal on your Render instance

### Step 3: Download File to Persistent Disk

In the Render shell:

```bash
# Navigate to persistent disk
cd /opt/render/project/data

# Download from Google Drive
pip install gdown
gdown "https://drive.google.com/uc?id=YOUR_FILE_ID_HERE"

# OR download from Dropbox
curl -L -o training_data.parquet "https://www.dropbox.com/s/YOUR_LINK/training_data.parquet?dl=1"

# Verify file exists
ls -lh training_data.parquet

# Check it's readable
python3 -c "import pandas as pd; df = pd.read_parquet('training_data.parquet'); print(f'Loaded {len(df)} rows')"
```

### Step 4: Update Code to Use Persistent Disk

The file is now at `/opt/render/project/data/training_data.parquet`

But your code looks for `data/training_data.parquet` (relative path).

**Solution:** Create a symlink in your app directory:

Add to Render **Start Command**:
```bash
ln -sf /opt/render/project/data/training_data.parquet data/training_data.parquet && python -m uvicorn api:app --host 0.0.0.0 --port $PORT
```

This creates a symbolic link so the code finds the file!

---

## METHOD 2: Upload via SCP (If You Have SSH Access)

If Render gives you SSH access:

```bash
# From your local machine
scp data/training_data.parquet render:/opt/render/project/data/
```

---

## METHOD 3: Download During Build (Automated)

### Create a build script that downloads to persistent disk

**File:** `setup_persistent_data.sh`
```bash
#!/bin/bash

# Persistent disk location
DISK_PATH="/opt/render/project/data"
FILE_PATH="${DISK_PATH}/training_data.parquet"

# Check if file already exists (don't re-download)
if [ -f "$FILE_PATH" ]; then
    echo "✅ Training data already exists on persistent disk"
    ln -sf "$FILE_PATH" data/training_data.parquet
    exit 0
fi

# Download file (only on first deploy)
echo "📥 Downloading training data to persistent disk..."
mkdir -p "$DISK_PATH"
pip install -q gdown

# Download from Google Drive
gdown "https://drive.google.com/uc?id=YOUR_FILE_ID" -O "$FILE_PATH"

# Create symlink
ln -sf "$FILE_PATH" data/training_data.parquet

echo "✅ Training data setup complete"
```

**Render Build Command:**
```bash
chmod +x setup_persistent_data.sh && ./setup_persistent_data.sh && pip install -r requirements.txt
```

**Benefits:**
- File downloads once
- Persists across deployments
- Automatic symlink creation

---

## METHOD 4: Use Render's File Upload (If Available)

Some Render plans allow direct file upload to persistent disks via dashboard. Check if your plan has this feature.

---

## 🎯 RECOMMENDED: Method 1 (Render Shell)

**Why:**
- Fastest (5 minutes)
- No code changes needed
- File persists forever
- Can verify immediately

**Steps:**
1. Upload file to Google Drive
2. Get file ID
3. Open Render Shell
4. Run: `cd /opt/render/project/data && pip install gdown && gdown "https://drive.google.com/uc?id=FILE_ID"`
5. Update start command to create symlink
6. Done!

---

## AFTER UPLOAD

### Verify It Works

In Render Shell:
```bash
# Check file exists
ls -lh /opt/render/project/data/training_data.parquet

# Test loading
cd /opt/render/project/src
python3 -c "
from technic_v4.engine.meta_experience import load_meta_experience
meta = load_meta_experience()
print(f'✅ Meta experience loaded: {meta is not None}')
if meta:
    print(f'   Score column: {meta.score_col}')
    print(f'   Horizons: {meta.horizons}')
    print(f'   Buckets: {len(meta.bucket_stats)}')
"
```

Should output:
```
✅ Meta experience loaded: True
   Score column: TechRating
   Horizons: [5, 10]
   Buckets: 10
```

---

## RESULT

✅ File on persistent disk (survives deployments)  
✅ Meta-experience feature fully working  
✅ No Git LFS issues  
✅ No features lost  
✅ Production-ready!
