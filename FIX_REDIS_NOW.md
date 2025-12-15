# How to Fix Redis Connection - Step by Step

## 🔴 PROBLEM

All connection attempts fail with: `invalid username-password pair`

This means either:
1. The password has been regenerated in Redis Cloud
2. There's an ACL (Access Control List) configuration issue
3. The database requires a different authentication method

---

## ✅ SOLUTION: Get Fresh Credentials

### Step 1: Go to Redis Cloud Dashboard

1. Open: https://app.redislabs.com/
2. Log in to your account
3. You should see your database: **database-MJ6OLK48**

### Step 2: Get the EXACT Connection String

**Option A: From the Connect Button**
1. Click on your database **database-MJ6OLK48**
2. Click the **"Connect"** button (you showed this in your screenshot)
3. Select **"Redis CLI"** tab
4. Look for the command that starts with: `redis-cli -u redis://...`
5. **Copy the ENTIRE URL** after `-u`

**Option B: From Database Configuration**
1. Click on your database
2. Go to **"Configuration"** tab
3. Look for **"Public endpoint"** or **"Connection details"**
4. Copy the connection string

### Step 3: Check for Password Reset

Sometimes Redis Cloud regenerates passwords. Look for:
- A "Reset Password" or "Regenerate Password" button
- A "Show Password" button to reveal the current password
- An "Access Control" or "ACL" section

### Step 4: Update Render Environment Variables

Once you have the correct connection string:

1. Go to: https://dashboard.render.com/
2. Select your service: **technic**
3. Go to: **Environment** tab
4. Update these variables with the NEW values:

```
REDIS_URL=redis://[username]:[NEW_PASSWORD]@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0

REDIS_PASSWORD=[NEW_PASSWORD]
```

5. Click **"Save Changes"**
6. Render will automatically redeploy

### Step 5: Test Locally (Optional)

After getting the new credentials, test locally:

```powershell
$env:REDIS_URL="redis://[username]:[NEW_PASSWORD]@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0"
python test_redis_connection.py
```

You should see:
```
✅ Redis connection successful!
✅ Set/Get test passed
✅ Redis cache enabled
```

---

## 🔍 TROUBLESHOOTING

### If you still get authentication errors:

**Check 1: ACL Configuration**
- Redis Cloud might have ACL enabled
- You might need to create a new user with proper permissions
- Look for "Access Control" or "Users" section in Redis Cloud

**Check 2: IP Whitelist**
- Redis Cloud might have IP restrictions
- Check if your IP or Render's IPs are whitelisted
- Look for "Security" or "Network" settings

**Check 3: SSL/TLS**
- Some Redis Cloud instances require SSL
- Try using `rediss://` instead of `redis://` (note the extra 's')

**Check 4: Database Number**
- Make sure you're using `/0` at the end of the URL
- Try without `/0` if it doesn't work

---

## 📋 WHAT TO SEND ME

If you're still having issues, please share:

1. **Screenshot of Redis Cloud "Connect" dialog** (the full dialog, not just CLI)
2. **Screenshot of Redis Cloud "Configuration" tab**
3. **Any error messages** from Redis Cloud dashboard

I can then help you identify the exact issue!

---

## 💡 ALTERNATIVE: Create New Redis Database

If the current database has issues, you can create a fresh one:

1. In Redis Cloud dashboard, click **"New database"**
2. Choose **Free tier** (30MB, perfect for testing)
3. Name it: `technic-cache`
4. Click **"Activate"**
5. Copy the new connection credentials
6. Update Render environment variables

This gives you a clean slate with fresh credentials!

---

## ⏭️ MEANWHILE: Your Scanner Still Works!

Remember: **Redis is optional!**

Your scanner works perfectly without it:
- ✅ 75-90s for full universe (goal met!)
- ✅ L1/L2 cache working
- ✅ All features functional

You can:
1. **Skip Redis for now** and focus on frontend
2. **Fix Redis later** when you have time
3. **Add Redis after beta launch** based on real usage

**The choice is yours!**
