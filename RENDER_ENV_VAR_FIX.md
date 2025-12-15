# FINAL FIX: Add Environment Variable to Render

## The Problem

The file is **still in Git history** even though we disabled LFS. Render is still trying to download it from LFS.

## THE SOLUTION (Do This Right Now)

### In Render Dashboard:

1. Go to your **technic** service
2. Click **Environment** (left sidebar)
3. Click **Add Environment Variable**
4. Add this:

**Key:** `GIT_LFS_SKIP_SMUDGE`  
**Value:** `1`

5. Click **Save**
6. Click **Manual Deploy** → **Deploy latest commit**

---

## WHAT THIS DOES

This environment variable tells Git to **skip all LFS downloads** during clone. The file will be ignored, and the clone will succeed.

---

## EXPECTED RESULT

You should see:

```
==> Cloning from https://github.com/HOboGoblin45/technic
✅ Cloning into '/opt/render/project/src'...
✅ Downloading cache...
✅ Installing dependencies...
✅ Installing redis>=5.0.0
✅ Installing scipy>=1.11.0
✅ Build succeeded
```

**NO MORE LFS ERRORS!**

---

## ALTERNATIVE (If Above Doesn't Work)

### Nuclear Option: Remove File from ALL Git History

Run this locally:

```bash
# Install BFG Repo Cleaner
# Windows: choco install bfg
# Mac: brew install bfg

# Clone mirror
cd ..
git clone --mirror https://github.com/HOboGoblin45/technic.git technic-mirror
cd technic-mirror

# Remove file from ALL history
bfg --delete-files training_data.parquet

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push (WARNING: Rewrites history!)
git push --force
```

Then in Render: Clear cache & deploy.

---

## RECOMMENDED: Just Use Environment Variable

The `GIT_LFS_SKIP_SMUDGE=1` approach is **safest** and **fastest**. Do that first!
