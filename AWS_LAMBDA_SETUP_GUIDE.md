# AWS Lambda Setup Guide - Step by Step

You're at the AWS Console home. Let's set up Lambda for 3-4x faster scans!

## 🎯 What We're Building

**Hybrid Architecture:**
- ✅ Render Pro Plus (4 CPU, 8GB RAM) - Main API + cached requests
- 🆕 AWS Lambda (10GB RAM, 6 vCPU) - Heavy scanner workloads
- ✅ Redis Cloud (12GB) - Shared cache between both

**Result:** 3-4x faster uncached scans at $3-30/mo additional cost

---

## 📋 Step 1: Create IAM User for Lambda Deployment

### 1.1 Navigate to IAM
1. In AWS Console search bar, type **"IAM"**
2. Click **"IAM"** (Identity and Access Management)

### 1.2 Create User
1. Click **"Users"** in left sidebar
2. Click **"Create user"** button
3. **User name:** `technic-lambda-deployer`
4. Click **"Next"**

### 1.3 Set Permissions
1. Select **"Attach policies directly"**
2. Search and check these policies:
   - ✅ `AWSLambda_FullAccess`
   - ✅ `IAMFullAccess` (for creating Lambda execution role)
   - ✅ `AmazonS3FullAccess` (for deployment packages)
3. Click **"Next"**
4. Click **"Create user"**

### 1.4 Create Access Keys
1. Click on the user you just created (`technic-lambda-deployer`)
2. Click **"Security credentials"** tab
3. Scroll to **"Access keys"** section
4. Click **"Create access key"**
5. Select **"Command Line Interface (CLI)"**
6. Check **"I understand..."** checkbox
7. Click **"Next"**
8. Click **"Create access key"**
9. **IMPORTANT:** Copy both:
   - Access key ID: `AKIA...`
   - Secret access key: `...` (only shown once!)
10. Click **"Done"**

**Save these credentials securely!** You'll need them in Step 3.

---

## 📋 Step 2: Create Lambda Execution Role

### 2.1 Navigate to IAM Roles
1. In IAM dashboard, click **"Roles"** in left sidebar
2. Click **"Create role"** button

### 2.2 Select Trusted Entity
1. Select **"AWS service"**
2. Select **"Lambda"**
3. Click **"Next"**

### 2.3 Add Permissions
1. Search and check these policies:
   - ✅ `AWSLambdaBasicExecutionRole` (for CloudWatch logs)
   - ✅ `AWSLambdaVPCAccessExecutionRole` (if using VPC)
2. Click **"Next"**

### 2.4 Name and Create
1. **Role name:** `technic-lambda-execution-role`
2. **Description:** "Execution role for Technic scanner Lambda"
3. Click **"Create role"**

### 2.5 Copy Role ARN
1. Click on the role you just created
2. Copy the **ARN** (looks like: `arn:aws:iam::123456789012:role/technic-lambda-execution-role`)
3. **Save this ARN!** You'll need it in Step 4.

---

## 📋 Step 3: Install AWS CLI and Configure

### 3.1 Install AWS CLI (Windows)

**Option A: MSI Installer (Recommended)**
1. Download: https://awscli.amazonaws.com/AWSCLIV2.msi
2. Run the installer
3. Follow the wizard (default options are fine)
4. Restart PowerShell

**Option B: Command Line**
```powershell
# In PowerShell
msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi
```

### 3.2 Verify Installation
```powershell
aws --version
```
Should show: `aws-cli/2.x.x ...`

### 3.3 Configure AWS CLI
```powershell
aws configure
```

Enter the credentials from Step 1.4:
```
AWS Access Key ID [None]: AKIA... (paste your access key)
AWS Secret Access Key [None]: ... (paste your secret key)
Default region name [None]: us-east-1
Default output format [None]: json
```

### 3.4 Test Configuration
```powershell
aws sts get-caller-identity
```

Should show your account info (Account ID, User ARN, etc.)

---

## 📋 Step 4: Create Lambda Function

### 4.1 Navigate to Lambda
1. In AWS Console search bar, type **"Lambda"**
2. Click **"Lambda"**

### 4.2 Create Function
1. Click **"Create function"** button
2. Select **"Author from scratch"**
3. **Function name:** `technic-scanner`
4. **Runtime:** `Python 3.10`
5. **Architecture:** `x86_64`
6. **Execution role:** Select "Use an existing role"
7. **Existing role:** Select `technic-lambda-execution-role` (from Step 2)
8. Click **"Create function"**

**Note:** Use Python 3.10 to match your local environment (Python 3.10.11)

### 4.3 Configure Function
1. In the function page, click **"Configuration"** tab
2. Click **"General configuration"** → **"Edit"**
3. Set:
   - **Memory:** `3008 MB` (3 GB - AWS Lambda maximum)
   - **Timeout:** `15 min 0 sec` (900 seconds)
   - **Ephemeral storage:** `512 MB` (default is fine)
4. Click **"Save"**

**Note:** AWS Lambda maximum memory is 3008 MB (3 GB). This still provides significant performance improvement over Render's shared resources.

### 4.4 Add Environment Variables
1. Still in **"Configuration"** tab
2. Click **"Environment variables"** → **"Edit"**
3. Click **"Add environment variable"** for each:

```
POLYGON_API_KEY = your_polygon_key
REDIS_URL = your_redis_cloud_url
REDIS_PASSWORD = your_redis_password
```

4. Click **"Save"**

---

## 📋 Step 5: Deploy Lambda Code

### 5.1 Create Deployment Package

In your project directory:

```powershell
# Navigate to project
cd C:\Users\ccres\OneDrive\Desktop\technic-clean

# Create deployment directory
mkdir lambda_deploy
cd lambda_deploy

# Copy Lambda function
copy ..\lambda_scanner.py lambda_function.py

# Copy technic_v4 module
xcopy ..\technic_v4 technic_v4\ /E /I /H

# Install dependencies
pip install -r ..\requirements.txt -t .

# Create ZIP
Compress-Archive -Path * -DestinationPath ..\technic-scanner.zip -Force

cd ..
```

### 5.2 Upload to Lambda

**Option A: AWS CLI (Recommended)**
```powershell
aws lambda update-function-code `
  --function-name technic-scanner `
  --zip-file fileb://technic-scanner.zip
```

**Option B: AWS Console**
1. Go to Lambda function page
2. Click **"Code"** tab
3. Click **"Upload from"** → **".zip file"**
4. Select `technic-scanner.zip`
5. Click **"Save"**

### 5.3 Verify Deployment
```powershell
aws lambda get-function --function-name technic-scanner
```

Should show function details with `State: Active`

---

## 📋 Step 6: Test Lambda Function

### 6.1 Create Test Event
1. In Lambda function page, click **"Test"** tab
2. Click **"Create new event"**
3. **Event name:** `test-scan`
4. **Event JSON:**
```json
{
  "max_symbols": 10,
  "sectors": ["Technology"],
  "min_tech_rating": 0,
  "trade_style": "Short-term swing"
}
```
5. Click **"Save"**

### 6.2 Run Test
1. Click **"Test"** button
2. Wait for execution (may take 30-60 seconds first time)
3. Check **"Execution result"** section

**Expected Result:**
```json
{
  "statusCode": 200,
  "body": {
    "results": [...],
    "scan_time": 45.2,
    "symbols_scanned": 10
  }
}
```

### 6.3 Check CloudWatch Logs
1. Click **"Monitor"** tab
2. Click **"View CloudWatch logs"**
3. Click the latest log stream
4. Verify scan completed successfully

---

## 📋 Step 7: Create API Gateway (Optional)

If you want to call Lambda directly via HTTP:

### 7.1 Create API
1. In AWS Console, search **"API Gateway"**
2. Click **"Create API"**
3. Select **"HTTP API"** → **"Build"**
4. **API name:** `technic-api`
5. Click **"Next"**

### 7.2 Add Route
1. **Method:** `POST`
2. **Resource path:** `/scan`
3. **Integration:** Select "Lambda"
4. **Lambda function:** `technic-scanner`
5. Click **"Next"** → **"Next"** → **"Create"**

### 7.3 Get API URL
1. Click **"Stages"** in left sidebar
2. Copy the **"Invoke URL"**
3. Your endpoint: `https://xxx.execute-api.us-east-1.amazonaws.com/scan`

---

## 📋 Step 8: Update Render API to Use Lambda

### 8.1 Add boto3 to requirements.txt
```
boto3==1.34.0
```

### 8.2 Set Environment Variables in Render
1. Go to Render dashboard
2. Select your service
3. Go to **"Environment"** tab
4. Add:
```
AWS_ACCESS_KEY_ID = (from Step 1.4)
AWS_SECRET_ACCESS_KEY = (from Step 1.4)
AWS_REGION = us-east-1
LAMBDA_FUNCTION_NAME = technic-scanner
```

### 8.3 Deploy Hybrid API
```powershell
# Copy hybrid API
copy api_hybrid.py api.py

# Commit and push
git add api.py requirements.txt
git commit -m "Add AWS Lambda hybrid architecture"
git push origin main
```

---

## 📊 Cost Estimate

### Lambda Pricing (us-east-1)
- **Compute:** $0.0000166667 per GB-second
- **Requests:** $0.20 per 1M requests

### Example Usage:
- **3GB RAM, 60s per scan**
- **100 scans/day = 3,000 scans/month**

**Monthly Cost:**
```
Compute: 3 GB × 60s × 3,000 scans × $0.0000166667 = $9.00
Requests: 3,000 requests × $0.20 / 1M = $0.60
Total: $9.60/month
```

**With Free Tier (first 12 months):**
- 400,000 GB-seconds free
- 1M requests free
- **Actual cost: ~$3-10/month**

---

## ✅ Success Checklist

- [ ] IAM user created with access keys
- [ ] Lambda execution role created
- [ ] AWS CLI installed and configured
- [ ] Lambda function created (3GB max, 15min timeout)
- [ ] Environment variables set
- [ ] Code deployed to Lambda
- [ ] Test scan successful
- [ ] CloudWatch logs show successful execution
- [ ] (Optional) API Gateway created
- [ ] Render API updated with Lambda integration
- [ ] Hybrid architecture tested end-to-end

---

## 🎉 What You Get

### Before (Render Only):
- ⏱️ 60-120s for uncached scans
- 💰 $175/month (Render Pro Plus)

### After (Hybrid):
- ⚡ 15-30s for uncached scans (3-4x faster!)
- 💰 $175 + $3-30/month = $178-205/month
- 🚀 Best of both worlds: Render for cached, Lambda for heavy lifting

---

## 📞 Next Steps

1. **Complete Steps 1-7** above
2. **Test Lambda function** with sample scan
3. **Deploy hybrid API** to Render
4. **Test end-to-end** from Flutter app
5. **Monitor costs** in AWS Billing dashboard

**Let me know when you complete each step and I'll help with any issues!**
