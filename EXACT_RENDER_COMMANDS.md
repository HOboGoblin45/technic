# EXACT Commands to Run in Render Shell

## Your Setup
- **Google Drive File ID:** `1OzKwpp5damY9rhS-11fmEVmXZIRI-EMm`
- **Persistent Disk:** `/opt/render/project/data` (5GB)
- **Current Issue:** Directory doesn't exist yet

---

## STEP-BY-STEP COMMANDS

### 1. Open Render Shell
In Render Dashboard → Your Service → Click "Shell" tab

### 2. Run These Commands (Copy/Paste)

```bash
# Create the data directory on persistent disk
mkdir -p /opt/render/project/src

# Navigate to it
cd /opt/render/project/data

# Install gdown
pip install gdown

# Download your file from Google Drive
gdown "https://drive.google.com/uc?id=1OzKwpp5damY9rhS-11fmEVmXZIRI-EMm"

# Verify file downloaded
ls -lh training_data_v2.parquet

# Check file is readable
python3 -c "import pandas as pd; df = pd.read_parquet('training_data.parquet'); print(f'✅ Loaded {len(df)} rows, {len(df.columns)} columns')"
```

### 3. Create Symlink in App Directory

```bash
# Navigate to app source
cd /opt/render/project/src

# Create data directory if it doesn't exist
mkdir -p data

# Create symlink to persistent disk file
ln -sf /opt/render/project/data/training_data_v2.parquet data/training_data_v2.parquet

# Verify symlink works
ls -lh data/training_data_v2.parquet
```

### 4. Test Meta Experience Loads

```bash
# Still in /opt/render/project/src
python3 -c "
from technic_v4.engine.meta_experience import load_meta_experience
meta = load_meta_experience()
if meta:
    print('✅ Meta experience loaded successfully!')
    print(f'   Score column: {meta.score_col}')
    print(f'   Horizons: {meta.horizons}')
    print(f'   Buckets: {len(meta.bucket_stats)} buckets')
else:
    print('❌ Meta experience not loaded')
"
```

---

## EXPECTED OUTPUT

```
✅ Loaded 50000 rows, 45 columns
✅ Meta experience loaded successfully!
   Score column: TechRating
   Horizons: [5, 10]
   Buckets: 10 buckets
```

---

## MAKE IT PERMANENT

### Update Render Start Command

In Render Dashboard → Settings → Start Command:

**Change from:**
```bash
python -m uvicorn api:app --host 0.0.0.0 --port $PORT
```

**To:**
```bash
mkdir -p data && ln -sf /opt/render/project/data/training_data.parquet data/training_data.parquet && python -m uvicorn api:app --host 0.0.0.0 --port $PORT
```

This ensures the symlink is created on every deployment!

---

## RESULT

✅ File on persistent disk (survives deployments)  
✅ Symlink created automatically on startup  
✅ Meta-experience feature fully working  
✅ All features preserved  
✅ No Git LFS issues  

**Your scanner will now have full meta-experience insights using your training data!**
