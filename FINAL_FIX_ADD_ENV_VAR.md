# FINAL FIX: Add Environment Variable to Render

## The Issue

The file `data/training_data.parquet` is **still in Git history** even though you disabled LFS in .gitattributes.

Render is still trying to download it from LFS during clone.

---

## THE SOLUTION (Do This Now)

### In Render Dashboard:

1. Go to your **technic** service
2. Click **Environment** (left sidebar)
3. Click **Add Environment Variable**
4. Add:

**Key:** `GIT_LFS_SKIP_SMUDGE`  
**Value:** `1`

5. Click **Save Changes**
6. Render will automatically redeploy

---

## WHAT THIS DOES

This tells Git to **skip all LFS downloads** during clone. The file in Git history will be ignored, and the clone will succeed.

**Your persistent disk file will still work** because you have the symlink in the start command!

---

## EXPECTED RESULT

After adding the environment variable, you should see:

```
==> Cloning from https://github.com/HOboGoblin45/technic
✅ Cloning into '/opt/render/project/src'...
✅ Downloading cache...
✅ Installing dependencies...
✅ Installing scipy>=1.11.0
✅ Installing redis>=5.0.0
✅ Build succeeded
✅ Starting service...
[META] loaded meta experience from data/training_data_v2.parquet
```

**NO MORE LFS ERRORS!**

---

## WHY THIS WORKS

1. **Git clone succeeds** (skips LFS download)
2. **Your start command runs** (creates symlink)
3. **Symlink points to persistent disk** (where you uploaded the file)
4. **Meta experience loads** from persistent disk
5. **Everything works!**

---

## SUMMARY

You have TWO copies of the data:
1. **In Git history** (broken LFS link) ← Skip this with env var
2. **On persistent disk** (working file) ← Use this via symlink

The environment variable makes Render ignore #1 and use #2!

---

## DO THIS NOW

1. Add `GIT_LFS_SKIP_SMUDGE=1` to Render environment variables
2. Save
3. Wait for automatic redeploy
4. ✅ Done!
