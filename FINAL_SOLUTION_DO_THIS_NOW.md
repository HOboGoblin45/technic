# FINAL SOLUTION - Do These 3 Steps

## The Issue

`data/training_data.parquet` is in Git history (commits 1482adb through ca8901c). Even though it's deleted in the latest commit, Render still tries to download it from LFS during checkout.

---

## SOLUTION (3 Steps)

### Step 1: Add Environment Variable in Render

1. Go to Render Dashboard → Your Service
2. Click **Environment** (left sidebar)
3. Click **Add Environment Variable**
4. Add:
   - **Key:** `GIT_LFS_SKIP_SMUDGE`
   - **Value:** `1`
5. **Save Changes** (don't deploy yet)

### Step 2: Clear Render Build Cache

1. Still in Render Dashboard
2. Click **Settings** (left sidebar)
3. Scroll to **Build & Deploy** section
4. Click **Clear build cache** button
5. Confirm

### Step 3: Update Start Command

1. Still in **Settings**
2. Find **Start Command**
3. Replace with:

```bash
mkdir -p data && ln -sf /opt/render/project/data/training_data_v2.parquet data/training_data_v2.parquet && python -m uvicorn api:app --host 0.0.0.0 --port $PORT
```

4. Click **Save Changes**
5. Render will automatically deploy

---

## EXPECTED RESULT

```
==> Downloading cache...
==> Cloning from https://github.com/HOboGoblin45/technic
✅ Cloning into '/opt/render/project/src'...
✅ Checkout succeeded (LFS files skipped)
==> Installing dependencies...
✅ Installing scipy>=1.11.0
✅ Installing redis>=5.0.0
✅ Build succeeded
==> Starting service...
✅ Creating symlink to persistent disk...
[META] loaded meta experience from data/training_data_v2.parquet
✅ Service started successfully
```

---

## WHY THIS WORKS

1. **`GIT_LFS_SKIP_SMUDGE=1`** - Tells Git to skip ALL LFS downloads
2. **Clear build cache** - Removes old cached LFS references
3. **Start command** - Creates symlink to your persistent disk file

**Result:** Git clone succeeds, symlink works, meta experience loads!

---

## DO ALL 3 STEPS NOW

The order matters:
1. Add environment variable
2. Clear cache
3. Update start command
4. ✅ Deploy succeeds!
