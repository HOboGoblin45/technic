# Upload Training Data to Render Persistent Disk

## ⚠️ Current Status

Your API is **working fine** without this file! The warning is non-critical:

```
⚠️  Warning: training_data_v2.parquet not found on persistent disk
```

**What works:**
- ✅ Scanner (main feature)
- ✅ Symbol details
- ✅ Copilot AI
- ✅ All API endpoints

**What doesn't work:**
- ❌ Meta experience feature (uses training_data_v2.parquet)

---

## 🎯 Do You Need to Fix This?

**Skip this if:**
- You don't use the meta experience feature
- You're just testing the scanner
- The file isn't critical for your app

**Fix this if:**
- You need the meta experience feature
- You want to eliminate the warning
- You use ML models that depend on this data

---

## 📤 How to Upload the File

### **Option 1: Upload via Render Shell (Recommended)**

1. **Go to Render Dashboard:**
   ```
   https://dashboard.render.com
   ```

2. **Click on "technic" service**

3. **Click "Shell" tab** (opens a terminal in your container)

4. **Check if file exists locally:**
   ```bash
   ls -lh data/training_data_v2.parquet
   ```

5. **If file exists locally, copy to persistent disk:**
   ```bash
   # Create directory on persistent disk
   mkdir -p /opt/render/project/data
   
   # Copy file
   cp data/training_data_v2.parquet /opt/render/project/data/
   
   # Verify
   ls -lh /opt/render/project/data/training_data_v2.parquet
   ```

6. **Restart service** (click "Manual Deploy" → "Clear build cache & deploy")

---

### **Option 2: Upload from Your Computer**

If the file isn't in your Git repo, you'll need to upload it:

#### **Step 1: Get the File**

The file should be in your local project:
```
c:/Users/ccres/OneDrive/Desktop/technic-clean/data/training_data_v2.parquet
```

#### **Step 2: Upload via Render Shell**

1. **Open Render Shell** (Dashboard → technic → Shell)

2. **Create a temporary upload script:**
   ```bash
   cat > /tmp/upload.sh << 'EOF'
   #!/bin/bash
   # This script will be used to receive the file
   mkdir -p /opt/render/project/data
   echo "Ready to receive file. Paste base64 content and press Ctrl+D"
   base64 -d > /opt/render/project/data/training_data_v2.parquet
   echo "File uploaded successfully!"
   ls -lh /opt/render/project/data/training_data_v2.parquet
   EOF
   
   chmod +x /tmp/upload.sh
   ```

3. **On your local machine, encode the file:**
   ```powershell
   # In PowerShell
   $bytes = [System.IO.File]::ReadAllBytes("c:/Users/ccres/OneDrive/Desktop/technic-clean/data/training_data_v2.parquet")
   $base64 = [System.Convert]::ToBase64String($bytes)
   $base64 | Out-File -FilePath "training_data_base64.txt" -Encoding ASCII
   ```

4. **Copy the base64 content and paste into Render Shell**

**Note:** This method is complex for large files (126MB). See Option 3 for easier approach.

---

### **Option 3: Use Cloud Storage (Easiest for Large Files)**

#### **Step 1: Upload to Google Drive/Dropbox**

1. Upload `training_data_v2.parquet` to Google Drive
2. Get a shareable link
3. Make it publicly accessible

#### **Step 2: Download in Render Shell**

1. **Open Render Shell**

2. **Download file:**
   ```bash
   # Create directory
   mkdir -p /opt/render/project/data
   
   # Download from Google Drive (replace FILE_ID)
   wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=FILE_ID' -O /opt/render/project/data/training_data_v2.parquet
   
   # Or use gdown (if installed)
   pip install gdown
   gdown https://drive.google.com/uc?id=FILE_ID -O /opt/render/project/data/training_data_v2.parquet
   
   # Verify
   ls -lh /opt/render/project/data/training_data_v2.parquet
   ```

3. **Restart service**

---

### **Option 4: Add to Git Repo (Not Recommended - 126MB)**

**Why not recommended:**
- File is 126MB (too large for Git)
- Will slow down clones
- Already removed from Git for this reason

**If you really want to:**
1. Add file back to `data/` directory
2. Commit and push
3. Render will include it in next deploy

---

## 🔧 Verify Upload

After uploading, verify in Render Shell:

```bash
# Check file exists
ls -lh /opt/render/project/data/training_data_v2.parquet

# Check file size (should be ~126MB)
du -h /opt/render/project/data/training_data_v2.parquet

# Test if Python can read it
python3 << EOF
import pandas as pd
df = pd.read_parquet('/opt/render/project/data/training_data_v2.parquet')
print(f"Loaded {len(df)} rows")
EOF
```

---

## 🎯 Recommended Approach

**For now: Skip it!**

Your API is working perfectly without this file. The meta experience is an optional feature.

**If you need it later:**
1. Use Option 3 (Cloud Storage) - easiest for 126MB file
2. Upload to Google Drive
3. Download in Render Shell
4. Restart service

---

## 📊 Summary

**Current Status:**
- ⚠️ Warning appears but **doesn't affect main functionality**
- ✅ Scanner works (your main feature)
- ✅ All API endpoints functional
- ❌ Meta experience feature unavailable

**To Fix:**
- Upload `training_data_v2.parquet` to `/opt/render/project/data/`
- Use cloud storage method for easiest upload
- Restart service after upload

**Recommendation:**
- **Don't fix it now** - focus on testing your working features
- Fix later if you need meta experience
- The warning is harmless

Your Technic app is fully functional! 🎉
