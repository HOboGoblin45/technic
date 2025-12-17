# Quick Instructions: Upload Lambda Package

## ✅ Once the ZIP is Created

You'll have a file called `technic-scanner.zip` in your project directory.

## 📤 Upload to AWS Lambda

### Method 1: AWS Console (If ZIP < 50MB)

1. Go to your Lambda function page: `technic-scanner`
2. Click the **"Code"** tab
3. Click **"Upload from"** → **".zip file"**
4. Select `technic-scanner.zip`
5. Click **"Save"**
6. Wait for upload to complete

### Method 2: AWS CLI (If ZIP > 50MB or Method 1 fails)

```powershell
aws lambda update-function-code `
  --function-name technic-scanner `
  --zip-file fileb://technic-scanner.zip
```

## ⚙️ Configure Lambda (In AWS Console)

### 1. Set Memory & Timeout
- Go to **Configuration** → **General configuration** → **Edit**
- Memory: `3008 MB`
- Timeout: `15 min 0 sec`
- Click **Save**

### 2. Add Environment Variables
- Go to **Configuration** → **Environment variables** → **Edit**
- Add these variables:
  - `POLYGON_API_KEY` = your_polygon_key
  - `REDIS_URL` = your_redis_cloud_url
  - `REDIS_PASSWORD` = your_redis_password
- Click **Save**

## 🧪 Test Lambda

### 1. Create Test Event
- Go to **Test** tab
- Click **"Create new event"**
- Event name: `test-scan`
- Event JSON:
```json
{
  "max_symbols": 10,
  "sectors": ["Technology"],
  "min_tech_rating": 0,
  "trade_style": "Short-term swing"
}
```
- Click **Save**

### 2. Run Test
- Click **"Test"** button
- Wait 30-60 seconds
- Check results

### 3. Check Logs
- Go to **Monitor** tab
- Click **"View CloudWatch logs"**
- Click latest log stream
- Verify scan completed

## 🎉 Expected Results

- **First run:** 30-60 seconds (cold start)
- **Subsequent runs:** 20-40 seconds
- **Status:** Should return scan results with symbols

## ⚠️ Troubleshooting

### If upload fails:
- Check ZIP size (should be < 250MB unzipped)
- Use AWS CLI method instead
- Check AWS credentials are configured

### If test fails:
- Check CloudWatch logs for errors
- Verify environment variables are set
- Check Lambda has correct execution role
- Verify Polygon API key is valid

### If timeout occurs:
- Increase timeout to 15 minutes
- Check if data fetching is working
- Verify Redis connection (optional)

## 📊 Monitor Performance

After successful test:
- Check execution time in logs
- Compare to Render performance
- Verify results are correct
- Check AWS costs in Billing dashboard

---

**Next:** Once Lambda is working, integrate with Render API using `api_hybrid.py`
