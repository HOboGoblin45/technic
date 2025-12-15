# Render Start Command Fix

## Problem
The Render deployment was failing with:
```
sh: 1: mkdir -p data && ln -sf /opt/render/project/data/training_data_v2.parquet data/training_data_v2.parquet && exec python -m uvicorn api:app --host 0.0.0.0 --port "$PORT": not found
```

This happens because the start command was trying to use shell operators (`&&`, `exec`) directly, which doesn't work in Render's command execution.

## Solution
Created a `start.sh` bash script that handles:
1. Creating the data directory
2. Creating symlink to persistent disk
3. Starting the API server

## Steps to Fix in Render Dashboard

### 1. Update Start Command
Go to your Render service settings and change the **Start Command** to:

```bash
bash start.sh
```

That's it! Just replace the entire start command with `bash start.sh`.

### 2. Verify Environment Variables
Make sure these are still set:
- `GIT_LFS_SKIP_SMUDGE=1` (to skip LFS downloads)
- Any other API keys (POLYGON_API_KEY, etc.)

### 3. Verify Persistent Disk
Ensure your persistent disk is:
- ✅ Mounted at `/opt/render/project/data`
- ✅ Contains `training_data_v2.parquet` (126MB file)

### 4. Deploy
After updating the start command, Render will automatically redeploy. The new deployment should:
1. ✅ Clone the repo (without LFS files)
2. ✅ Build the Docker image
3. ✅ Run `bash start.sh`
4. ✅ Create symlink to persistent disk
5. ✅ Start the API server successfully

## What the Script Does

```bash
#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Create symlink to training data on persistent disk
if [ -f "/opt/render/project/data/training_data_v2.parquet" ]; then
    ln -sf /opt/render/project/data/training_data_v2.parquet data/training_data_v2.parquet
    echo "✅ Symlink created for training_data_v2.parquet"
else
    echo "⚠️  Warning: training_data_v2.parquet not found on persistent disk"
fi

# Start the API server
exec python -m uvicorn api:app --host 0.0.0.0 --port "$PORT"
```

## Expected Result
After deployment, you should see in the logs:
```
✅ Symlink created for training_data_v2.parquet
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:10000
```

## Troubleshooting

### If deployment still fails:
1. Check that `start.sh` is in the repo root
2. Verify the file has Unix line endings (LF, not CRLF)
3. Check Render logs for specific error messages

### If symlink fails:
1. Verify persistent disk is mounted correctly
2. Check that `training_data_v2.parquet` exists on persistent disk
3. SSH into Render and manually verify: `ls -la /opt/render/project/data/`

### If API doesn't start:
1. Check that all Python dependencies are installed
2. Verify `api.py` exists in repo root
3. Check for any import errors in logs

## Summary
✅ Created `start.sh` script
✅ Pushed to GitHub
✅ Ready to update Render start command to: `bash start.sh`

This fix eliminates the shell syntax issues and provides better error handling and logging.
