# Simple Lambda Upload Script
# Run this after ZIP creation completes

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Upload Lambda Package to AWS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if ZIP exists
if (-not (Test-Path "technic-scanner.zip")) {
    Write-Host "ERROR: technic-scanner.zip not found!" -ForegroundColor Red
    Write-Host "Please wait for ZIP creation to complete first." -ForegroundColor Yellow
    exit 1
}

# Get ZIP size
$zipSize = (Get-Item technic-scanner.zip).Length / 1MB
Write-Host "Package size: $([math]::Round($zipSize, 2)) MB" -ForegroundColor Cyan
Write-Host ""

# Check if AWS CLI is available
try {
    $awsVersion = aws --version 2>&1
    Write-Host "AWS CLI found: $awsVersion" -ForegroundColor Green
    Write-Host ""
    
    # Upload to Lambda
    Write-Host "Uploading to Lambda function: technic-scanner..." -ForegroundColor Yellow
    Write-Host "This may take 3-5 minutes..." -ForegroundColor Gray
    Write-Host ""
    
    aws lambda update-function-code `
        --function-name technic-scanner `
        --zip-file fileb://technic-scanner.zip
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "Upload Successful!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next Steps:" -ForegroundColor Yellow
        Write-Host "1. Go to AWS Lambda Console" -ForegroundColor White
        Write-Host "2. Open 'technic-scanner' function" -ForegroundColor White
        Write-Host "3. Click 'Test' button" -ForegroundColor White
        Write-Host "4. Check CloudWatch logs for:" -ForegroundColor White
        Write-Host "   - 'Connected to Redis Cloud'" -ForegroundColor Gray
        Write-Host "   - Scan completion message" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Expected Results:" -ForegroundColor Yellow
        Write-Host "- First test: 30-60 seconds" -ForegroundColor White
        Write-Host "- Second test: 1-2 seconds (Redis cache)" -ForegroundColor White
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "Upload failed!" -ForegroundColor Red
        Write-Host "Try uploading via AWS Console instead:" -ForegroundColor Yellow
        Write-Host "1. Go to Lambda function page" -ForegroundColor White
        Write-Host "2. Click 'Upload from' -> '.zip file'" -ForegroundColor White
        Write-Host "3. Select technic-scanner.zip" -ForegroundColor White
        Write-Host "4. Click 'Save'" -ForegroundColor White
    }
    
} catch {
    Write-Host "AWS CLI not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Upload via AWS Console:" -ForegroundColor Yellow
    Write-Host "1. Go to https://console.aws.amazon.com/lambda" -ForegroundColor White
    Write-Host "2. Open 'technic-scanner' function" -ForegroundColor White
    Write-Host "3. Click 'Upload from' -> '.zip file'" -ForegroundColor White
    Write-Host "4. Select technic-scanner.zip ($([math]::Round($zipSize, 2)) MB)" -ForegroundColor White
    Write-Host "5. Click 'Save'" -ForegroundColor White
    Write-Host "6. Wait 2-3 minutes for upload" -ForegroundColor White
    Write-Host ""
}
