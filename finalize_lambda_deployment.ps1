# Finalize Lambda Deployment Script
# This script verifies dependencies, creates ZIP, and uploads to Lambda

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Lambda Deployment Finalization Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Verify Redis is installed
Write-Host "[1/5] Verifying Redis installation..." -ForegroundColor Yellow
if (Test-Path "lambda_deploy\redis") {
    Write-Host "  ✓ Redis library found" -ForegroundColor Green
} else {
    Write-Host "  ✗ Redis library not found!" -ForegroundColor Red
    Write-Host "  Please run: pip install redis==5.0.0 -t lambda_deploy" -ForegroundColor Red
    exit 1
}

# Step 2: Check package size
Write-Host ""
Write-Host "[2/5] Checking package size..." -ForegroundColor Yellow
$deploySize = (Get-ChildItem lambda_deploy -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "  Package size: $([math]::Round($deploySize, 2)) MB" -ForegroundColor Cyan

if ($deploySize -gt 250) {
    Write-Host "  ⚠ Warning: Package exceeds 250MB unzipped limit" -ForegroundColor Yellow
    Write-Host "  Consider removing test files or using Lambda layers" -ForegroundColor Yellow
}

# Step 3: Remove old ZIP and create new one
Write-Host ""
Write-Host "[3/5] Creating deployment package..." -ForegroundColor Yellow

if (Test-Path "technic-scanner.zip") {
    Write-Host "  Removing old ZIP file..." -ForegroundColor Gray
    Remove-Item "technic-scanner.zip" -Force
}

Write-Host "  Creating new ZIP with dependencies..." -ForegroundColor Gray
Write-Host "  This may take 2-3 minutes..." -ForegroundColor Gray

try {
    Compress-Archive -Path lambda_deploy\* -DestinationPath technic-scanner.zip -CompressionLevel Optimal
    $zipSize = (Get-Item technic-scanner.zip).Length / 1MB
    Write-Host "  ✓ ZIP created: $([math]::Round($zipSize, 2)) MB" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Failed to create ZIP: $_" -ForegroundColor Red
    exit 1
}

# Step 4: Check if AWS CLI is available
Write-Host ""
Write-Host "[4/5] Checking AWS CLI..." -ForegroundColor Yellow

$awsCliAvailable = $false
try {
    $awsVersion = aws --version 2>&1
    Write-Host "  ✓ AWS CLI found: $awsVersion" -ForegroundColor Green
    $awsCliAvailable = $true
} catch {
    Write-Host "  ✗ AWS CLI not found" -ForegroundColor Yellow
    Write-Host "  You'll need to upload manually via AWS Console or install AWS CLI" -ForegroundColor Yellow
}

# Step 5: Upload to Lambda (if AWS CLI available)
Write-Host ""
Write-Host "[5/5] Uploading to Lambda..." -ForegroundColor Yellow

if ($awsCliAvailable) {
    Write-Host "  Uploading technic-scanner.zip to Lambda..." -ForegroundColor Gray
    Write-Host "  This may take 3-5 minutes..." -ForegroundColor Gray
    
    try {
        $uploadResult = aws lambda update-function-code `
            --function-name technic-scanner `
            --zip-file fileb://technic-scanner.zip 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ Upload successful!" -ForegroundColor Green
            Write-Host ""
            Write-Host "========================================" -ForegroundColor Cyan
            Write-Host "Deployment Complete!" -ForegroundColor Green
            Write-Host "========================================" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "Next Steps:" -ForegroundColor Yellow
            Write-Host "1. Go to AWS Lambda Console" -ForegroundColor White
            Write-Host "2. Click 'Test' button" -ForegroundColor White
            Write-Host "3. Verify Redis connection in logs" -ForegroundColor White
            Write-Host "4. Check scan completes successfully" -ForegroundColor White
            Write-Host ""
            Write-Host "Expected Results:" -ForegroundColor Yellow
            Write-Host "- First run: 30-60 seconds" -ForegroundColor White
            Write-Host "- Second run: 1-2 seconds (Redis cache)" -ForegroundColor White
            Write-Host "- CloudWatch logs show: 'Connected to Redis Cloud'" -ForegroundColor White
        } else {
            Write-Host "  ✗ Upload failed!" -ForegroundColor Red
            Write-Host "  Error: $uploadResult" -ForegroundColor Red
            Write-Host ""
            Write-Host "Alternative: Upload via S3" -ForegroundColor Yellow
            Write-Host "  Run: .\upload_lambda_via_s3.ps1" -ForegroundColor Cyan
        }
    } catch {
        Write-Host "  ✗ Upload error: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "Alternative: Upload via AWS Console" -ForegroundColor Yellow
        Write-Host "1. Go to Lambda function page" -ForegroundColor White
        Write-Host "2. Click 'Upload from' -> '.zip file'" -ForegroundColor White
        Write-Host "3. Select technic-scanner.zip" -ForegroundColor White
        Write-Host "4. Click 'Save'" -ForegroundColor White
    }
} else {
    Write-Host "  Skipping upload (AWS CLI not available)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Package Ready for Manual Upload" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "File: technic-scanner.zip ($([math]::Round($zipSize, 2)) MB)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Upload Options:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Option 1: AWS Console (if ZIP < 50MB)" -ForegroundColor White
    Write-Host "1. Go to Lambda function page" -ForegroundColor Gray
    Write-Host "2. Click 'Upload from' -> '.zip file'" -ForegroundColor Gray
    Write-Host "3. Select technic-scanner.zip" -ForegroundColor Gray
    Write-Host "4. Click 'Save'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Option 2: S3 Upload (recommended for large files)" -ForegroundColor White
    Write-Host "  Run: .\upload_lambda_via_s3.ps1" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Option 3: Install AWS CLI and run this script again" -ForegroundColor White
    Write-Host "  Run: .\install_aws_cli.ps1" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✓ Dependencies installed (including Redis)" -ForegroundColor Green
Write-Host "✓ Deployment package created" -ForegroundColor Green
Write-Host "  Package size: $([math]::Round($zipSize, 2)) MB" -ForegroundColor Cyan
Write-Host ""

if ($awsCliAvailable -and $LASTEXITCODE -eq 0) {
    Write-Host "✓ Uploaded to Lambda" -ForegroundColor Green
    Write-Host ""
    Write-Host "Ready to test!" -ForegroundColor Green
} else {
    Write-Host "⏳ Ready for upload" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Upload the package using one of the options above" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "For detailed testing instructions, see:" -ForegroundColor Cyan
Write-Host "  LAMBDA_TESTING_AND_RENDER_INTEGRATION.md" -ForegroundColor White
Write-Host ""
