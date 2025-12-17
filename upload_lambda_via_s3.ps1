# Upload Lambda Package via S3
# This script uploads the large Lambda package to S3, then updates Lambda from S3

$ErrorActionPreference = "Stop"
$awsCmd = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Lambda Upload via S3" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$bucketName = "technic-lambda-deploy-$(Get-Random -Maximum 9999)"
$zipFile = "technic-scanner.zip"
$functionName = "technic-scanner"
$region = "us-east-1"

# Check if ZIP exists
if (-not (Test-Path $zipFile)) {
    Write-Host "ERROR: $zipFile not found!" -ForegroundColor Red
    Write-Host "Please run deploy_lambda.ps1 first to create the package." -ForegroundColor Yellow
    exit 1
}

$zipSize = (Get-Item $zipFile).Length / 1MB
Write-Host "Package size: $([math]::Round($zipSize, 2)) MB" -ForegroundColor White
Write-Host ""

# Step 1: Create S3 bucket
Write-Host "[1/4] Creating S3 bucket: $bucketName" -ForegroundColor Yellow
try {
    & $awsCmd s3 mb "s3://$bucketName" --region $region 2>&1 | Out-Null
    Write-Host "  OK Bucket created" -ForegroundColor Green
} catch {
    Write-Host "  WARNING: Bucket creation failed, it may already exist" -ForegroundColor Yellow
}

# Step 2: Upload ZIP to S3
Write-Host "[2/4] Uploading $zipFile to S3..." -ForegroundColor Yellow
Write-Host "  This may take 2-5 minutes depending on your internet speed..." -ForegroundColor Gray
try {
    & $awsCmd s3 cp $zipFile "s3://$bucketName/$zipFile" --region $region
    Write-Host "  OK Upload complete" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Upload failed: $_" -ForegroundColor Red
    exit 1
}

# Step 3: Update Lambda from S3
Write-Host "[3/4] Updating Lambda function from S3..." -ForegroundColor Yellow
try {
    & $awsCmd lambda update-function-code `
        --function-name $functionName `
        --s3-bucket $bucketName `
        --s3-key $zipFile `
        --region $region `
        --output json | ConvertFrom-Json | Format-List FunctionName, CodeSize, LastModified, Runtime, Timeout, MemorySize
    Write-Host "  OK Lambda updated successfully" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Lambda update failed: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Please check:" -ForegroundColor Yellow
    Write-Host "    1. Lambda function 'technic-scanner' exists" -ForegroundColor White
    Write-Host "    2. You have permission to update it" -ForegroundColor White
    Write-Host "    3. AWS credentials are correct" -ForegroundColor White
    exit 1
}

# Step 4: Clean up S3 bucket (optional)
Write-Host "[4/4] Cleaning up..." -ForegroundColor Yellow
Write-Host "  Do you want to delete the S3 bucket? (Y/N)" -ForegroundColor Yellow
$response = Read-Host "  "
if ($response -eq 'Y' -or $response -eq 'y') {
    try {
        & $awsCmd s3 rb "s3://$bucketName" --force --region $region 2>&1 | Out-Null
        Write-Host "  OK S3 bucket deleted" -ForegroundColor Green
    } catch {
        Write-Host "  WARNING: Could not delete bucket, you can delete it manually later" -ForegroundColor Yellow
    }
} else {
    Write-Host "  OK Keeping S3 bucket: $bucketName" -ForegroundColor White
    Write-Host "  You can delete it manually later if needed" -ForegroundColor Gray
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Lambda Upload Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Configure Lambda settings (memory, timeout, env vars)" -ForegroundColor White
Write-Host "  2. Test the function in AWS Console" -ForegroundColor White
Write-Host "  3. Check CloudWatch logs" -ForegroundColor White
Write-Host ""
Write-Host "See AWS_LAMBDA_SETUP_GUIDE.md for detailed instructions." -ForegroundColor Gray
Write-Host ""
