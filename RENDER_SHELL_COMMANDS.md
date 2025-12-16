# Render Shell Commands - Upload Training Data

## 📋 Copy and Paste These Commands

### **Step 1: Open Render Shell**
1. Go to https://dashboard.render.com
2. Click on "technic" service
3. Click "Shell" tab
4. Wait for shell to load

---

### **Step 2: Paste These Commands**

Copy and paste this entire block into Render Shell:

```bash
# Install gdown for Google Drive downloads
pip install gdown

# Create directory on persistent disk
mkdir -p /opt/render/project/data

# Download file from Google Drive
gdown 1OzKwpp5damY9rhS-11fmEVmXZIRI-EMm -O /opt/render/project/data/training_data_v2.parquet

# Verify file was downloaded
ls -lh /opt/render/project/data/training_data_v2.parquet

# Check file size
du -h /opt/render/project/data/training_data_v2.parquet

# Test if file is readable
python3 << 'EOF'
import pandas as pd
try:
    df = pd.read_parquet('/opt/render/project/data/training_data_v2.parquet')
    print(f"✅ SUCCESS! Loaded {len(df)} rows, {len(df.columns)} columns")
except Exception as e:
    print(f"❌ ERROR: {e}")
EOF

echo "✅ Upload complete! Restart your service to clear the warning."
```

---

### **Step 3: Restart Service**

After the commands complete successfully:

1. Go back to Render Dashboard
2. Click "Manual Deploy" button
3. Select "Clear build cache & deploy"
4. Wait for deployment to complete (~30-60 seconds)

---

## ✅ Expected Output

You should see:

```
Successfully installed gdown-x.x.x
Downloading...
From: https://drive.google.com/uc?id=1OzKwpp5damY9rhS-11fmEVmXZIRI-EMm
To: /opt/render/project/data/training_data_v2.parquet
100%|████████████████████████████████| 126M/126M [00:XX<00:00, XXMiB/s]

-rw-r--r-- 1 render render 126M Jan 15 12:34 /opt/render/project/data/training_data_v2.parquet

126M    /opt/render/project/data/training_data_v2.parquet

✅ SUCCESS! Loaded XXXX rows, XX columns

✅ Upload complete! Restart your service to clear the warning.
```

---

## 🔧 Troubleshooting

### **If gdown fails:**

Try wget instead:

```bash
# Alternative method using wget
mkdir -p /opt/render/project/data

wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1OzKwpp5damY9rhS-11fmEVmXZIRI-EMm' -O /opt/render/project/data/training_data_v2.parquet

# Verify
ls -lh /opt/render/project/data/training_data_v2.parquet
```

### **If file is too large for direct download:**

Google Drive may show a virus scan warning for large files. Use this:

```bash
# For large files with virus scan warning
pip install gdown
gdown --fuzzy 'https://drive.google.com/file/d/1OzKwpp5damY9rhS-11fmEVmXZIRI-EMm/view?usp=sharing' -O /opt/render/project/data/training_data_v2.parquet
```

---

## 📊 After Upload

Once uploaded and service restarted:

1. ✅ Warning will disappear
2. ✅ Meta experience feature will work
3. ✅ All features fully functional

---

## 🎯 Quick Reference

**File ID:** `1OzKwpp5damY9rhS-11fmEVmXZIRI-EMm`

**Destination:** `/opt/render/project/data/training_data_v2.parquet`

**Expected Size:** ~126MB

**Action After Upload:** Restart service via "Manual Deploy"
