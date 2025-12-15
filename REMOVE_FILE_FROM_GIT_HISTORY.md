# Remove training_data.parquet from Git History (Nuclear Option)

## The Problem

The file is in Git history, so even with `GIT_LFS_SKIP_SMUDGE=1`, Render's cache still has the old LFS reference.

## THE SOLUTION

Remove the file from ALL Git history using BFG Repo Cleaner.

---

## STEP-BY-STEP INSTRUCTIONS

### Step 1: Install BFG Repo Cleaner

**Windows (PowerShell as Administrator):**
```powershell
choco install bfg
```

**Or download manually:**
- Go to: https://rtyley.github.io/bfg-repo-cleaner/
- Download `bfg-1.14.0.jar`
- Save to your Desktop

### Step 2: Clone a Mirror of Your Repo

```bash
cd C:\Users\ccres\OneDrive\Desktop
git clone --mirror https://github.com/HOboGoblin45/technic.git technic-mirror
cd technic-mirror
```

### Step 3: Remove the File from ALL History

**If you installed via choco:**
```bash
bfg --delete-files training_data.parquet
```

**If you downloaded the JAR:**
```bash
java -jar C:\Users\ccres\Desktop\bfg-1.14.0.jar --delete-files training_data.parquet
```

### Step 4: Clean Up Git

```bash
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### Step 5: Force Push (Rewrites History!)

```bash
git push --force
```

### Step 6: Clean Up Local Repo

```bash
cd C:\Users\ccres\OneDrive\Desktop\technic-clean
git pull --force
```

### Step 7: Clear Render Cache

In Render Dashboard:
1. Go to your service
2. Click **Settings**
3. Scroll to **Build & Deploy**
4. Click **Clear build cache**
5. Click **Manual Deploy** → **Deploy latest commit**

---

## EXPECTED RESULT

```
==> Cloning from https://github.com/HOboGoblin45/technic
✅ Cloning into '/opt/render/project/src'...
✅ No LFS files to download
✅ Installing dependencies...
✅ Build succeeded
```

---

## ALTERNATIVE (If You Don't Want to Rewrite History)

### Just Delete the File from Current Commit

```bash
cd C:\Users\ccres\OneDrive\Desktop\technic-clean

# Remove file completely
git rm data/training_data.parquet
git commit -m "Remove training_data.parquet completely"
git push origin main
```

Then in Render:
1. Clear build cache
2. Add `GIT_LFS_SKIP_SMUDGE=1` environment variable
3. Deploy

---

## RECOMMENDATION

**Use the ALTERNATIVE (simpler):**
1. Delete file from current commit
2. Clear Render cache
3. Deploy

The file is already on your persistent disk, so you don't need it in Git at all!

---

## AFTER THIS WORKS

Your start command will create the symlink:
```bash
mkdir -p data && ln -sf /opt/render/project/data/training_data_v2.parquet data/training_data_v2.parquet && python -m uvicorn api:app --host 0.0.0.0 --port $PORT
```

And everything will work perfectly!
