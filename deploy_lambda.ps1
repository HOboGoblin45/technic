# Deploy Lambda Package Script
# This creates a deployment package for AWS Lambda

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AWS Lambda Deployment Package Creator" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Clean up old deployment
Write-Host "[1/6] Cleaning up old deployment..." -ForegroundColor Yellow
if (Test-Path "lambda_deploy") {
    Remove-Item -Path "lambda_deploy" -Recurse -Force
    Write-Host "  OK Removed old lambda_deploy directory" -ForegroundColor Green
}
if (Test-Path "technic-scanner.zip") {
    Remove-Item -Path "technic-scanner.zip" -Force
    Write-Host "  OK Removed old technic-scanner.zip" -ForegroundColor Green
}

# Step 2: Create deployment directory
Write-Host "[2/6] Creating deployment directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "lambda_deploy" | Out-Null
Write-Host "  OK Created lambda_deploy directory" -ForegroundColor Green

# Step 3: Copy Lambda function
Write-Host "[3/6] Copying Lambda function..." -ForegroundColor Yellow
Copy-Item "lambda_scanner.py" "lambda_deploy\lambda_function.py"
Write-Host "  OK Copied lambda_scanner.py to lambda_function.py" -ForegroundColor Green

# Step 4: Copy technic_v4 module
Write-Host "[4/6] Copying technic_v4 module..." -ForegroundColor Yellow
Copy-Item "technic_v4" "lambda_deploy\technic_v4" -Recurse
Write-Host "  OK Copied technic_v4 module" -ForegroundColor Green

# Step 5: Install dependencies
Write-Host "[5/6] Installing Lambda dependencies..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes..." -ForegroundColor Gray

# Use Lambda-specific requirements
pip install -r requirements_lambda.txt -t lambda_deploy --quiet --no-warn-script-location 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "  WARNING: Some dependencies may have issues" -ForegroundColor Yellow
    Write-Host "  Continuing anyway..." -ForegroundColor Yellow
}

# Step 6: Create ZIP file
Write-Host "[6/6] Creating deployment ZIP..." -ForegroundColor Yellow
Set-Location lambda_deploy
Compress-Archive -Path * -DestinationPath ..\technic-scanner.zip -Force
Set-Location ..

$zipSize = (Get-Item "technic-scanner.zip").Length / 1MB
$zipSizeRounded = [math]::Round($zipSize, 2)
Write-Host "  OK Created technic-scanner.zip ($zipSizeRounded MB)" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Deployment Package Ready!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Package: technic-scanner.zip" -ForegroundColor White
Write-Host "Size: $zipSizeRounded MB" -ForegroundColor White
Write-Host ""

if ($zipSize -gt 50) {
    Write-Host "WARNING: Package is larger than 50MB" -ForegroundColor Yellow
    Write-Host "  You will need to upload via S3 or use AWS CLI" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Upload command:" -ForegroundColor Cyan
    Write-Host "  aws lambda update-function-code --function-name technic-scanner --zip-file fileb://technic-scanner.zip" -ForegroundColor White
} else {
    Write-Host "OK Package size is good for direct upload" -ForegroundColor Green
    Write-Host ""
    Write-Host "Upload options:" -ForegroundColor Cyan
    Write-Host "  1. AWS Console: Upload .zip file in Lambda function page" -ForegroundColor White
    Write-Host "  2. AWS CLI: aws lambda update-function-code --function-name technic-scanner --zip-file fileb://technic-scanner.zip" -ForegroundColor White
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Upload technic-scanner.zip to AWS Lambda" -ForegroundColor White
Write-Host "  2. Test the function in AWS Console" -ForegroundColor White
Write-Host "  3. Check CloudWatch logs for any errors" -ForegroundColor White
Write-Host ""
