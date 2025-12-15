# URGENT: Clear Render Build Cache

## The Problem

Even though you removed the file from Git, **Render's cache still has the old clone** with the Git LFS error.

The error you're seeing:
```
Error downloading data/training_data.parquet (a11f61e)
Smudge error: Error downloading data/training_data.parquet
```

This is from Render's **cached Git clone**, not your current code!

---

## THE FIX (Do This Right Now)

### Step 1: Go to Render Dashboard

1. Open https://dashboard.render.com
2. Click on your **technic** service

### Step 2: Clear Build Cache

1. Click **Settings** (left sidebar)
2. Scroll down to **"Danger Zone"** section
3. Find **"Clear build cache"** button
4. Click it
5. Confirm when prompted

### Step 3: Manual Deploy

1. Go back to your service overview
2. Click **"Manual Deploy"** button (top right)
3. Select **"Clear build cache & deploy"**
4. Click **"Deploy"**

---

## WHAT THIS DOES

- **Deletes** Render's cached Git clone
- **Forces** a fresh clone from GitHub
- **Gets** your latest code (without the parquet file)
- **Builds** successfully

---

## EXPECTED RESULT

You should see in the logs:

```
✅ Cloning from https://github.com/HOboGoblin45/technic
✅ Downloading cache...
✅ Installing dependencies...
✅ Installing redis>=5.0.0
✅ Installing hiredis>=2.2.0
✅ Build succeeded
✅ Starting service...
```

**NO MORE Git LFS errors!**

---

## IF IT STILL FAILS

Try this nuclear option:

### Delete and Recreate the Service

1. In Render Dashboard → Settings
2. Scroll to bottom → "Delete Web Service"
3. Confirm deletion
4. Create new service:
   - Connect to GitHub repo
   - Select `main` branch
   - Use same settings
   - Add Redis environment variables

This guarantees a completely fresh start.

---

## VERIFICATION

After successful deployment, test Redis:

```bash
# SSH into Render (if available) or check logs
python test_redis_connection.py
```

Expected:
```
✅ Redis connection successful!
✅ Set/Get test passed
```

---

## WHY THIS IS NECESSARY

Render caches the Git clone for speed. But when there's a Git error in the cache, it keeps failing even after you fix the code. **Clearing the cache forces a fresh clone.**

---

## DO THIS NOW

1. Render Dashboard → Your Service
2. Settings → Clear build cache
3. Manual Deploy → Clear build cache & deploy
4. Watch it succeed!
