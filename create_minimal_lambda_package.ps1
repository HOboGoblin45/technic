# Create Minimal Lambda Package (Under 250MB unzipped)
# This installs only essential dependencies without dev/test files

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Creating Minimal Lambda Package" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Clean up old deployment
Write-Host "`n[1/5] Cleaning up old deployment..." -ForegroundColor Yellow
if (Test-Path "lambda_deploy_minimal") {
    Remove-Item -Recurse -Force lambda_deploy_minimal
}
New-Item -ItemType Directory -Force -Path lambda_deploy_minimal | Out-Null

# Copy core files
Write-Host "[2/5] Copying core Lambda files..." -ForegroundColor Yellow
Copy-Item lambda_deploy/lambda_function.py lambda_deploy_minimal/
Copy-Item -Recurse lambda_deploy/technic_v4 lambda_deploy_minimal/

# Install minimal dependencies (no tests, docs, or unnecessary files)
Write-Host "[3/5] Installing minimal dependencies..." -ForegroundColor Yellow
Write-Host "  This will take 2-3 minutes..." -ForegroundColor Gray

pip install --target lambda_deploy_minimal `
    --platform manylinux2014_x86_64 `
    --implementation cp `
    --python-version 3.11 `
    --only-binary=:all: `
    --upgrade `
    --no-cache-dir `
    redis==5.0.0 `
    numpy==1.24.3 `
    pandas==2.0.3 `
    scipy==1.11.3 `
    scikit-learn==1.3.0 `
    requests==2.31.0 `
    polygon-api-client==1.12.5 2>&1 | Out-Null

# Remove unnecessary files to reduce size
Write-Host "[4/5] Removing unnecessary files..." -ForegroundColor Yellow
$itemsToRemove = @(
    "*.dist-info",
    "*.egg-info", 
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "tests",
    "test",
    "*.so.debug",
    "*.pyx",
    "*.pxd",
    "*.c",
    "*.cpp",
    "*.h"
)

foreach ($pattern in $itemsToRemove) {
    Get-ChildItem -Path lambda_deploy_minimal -Recurse -Filter $pattern -Force -ErrorAction SilentlyContinue | 
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

# Check size
$size = (Get-ChildItem -Path lambda_deploy_minimal -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "  Unzipped size: $([math]::Round($size, 2)) MB" -ForegroundColor Green

if ($size -gt 250) {
    Write-Host "  WARNING: Still over 250MB limit!" -ForegroundColor Red
    Write-Host "  Trying aggressive cleanup..." -ForegroundColor Yellow
    
    # Remove more files
    Get-ChildItem -Path lambda_deploy_minimal -Recurse -Include "*.md","*.txt","*.rst" -Force | Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path lambda_deploy_minimal -Recurse -Directory -Filter "examples" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path lambda_deploy_minimal -Recurse -Directory -Filter "docs" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    
    $size = (Get-ChildItem -Path lambda_deploy_minimal -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host "  New unzipped size: $([math]::Round($size, 2)) MB" -ForegroundColor Green
}

# Create ZIP
Write-Host "[5/5] Creating ZIP file..." -ForegroundColor Yellow
if (Test-Path "technic-scanner-minimal.zip") {
    Remove-Item technic-scanner-minimal.zip -Force
}

Compress-Archive -Path lambda_deploy_minimal\* -DestinationPath technic-scanner-minimal.zip -CompressionLevel Optimal

$zipSize = (Get-Item technic-scanner-minimal.zip).Length / 1MB
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "SUCCESS!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Unzipped size: $([math]::Round($size, 2)) MB" -ForegroundColor Cyan
Write-Host "ZIP size: $([math]::Round($zipSize, 2)) MB" -ForegroundColor Cyan
Write-Host "`nPackage: technic-scanner-minimal.zip" -ForegroundColor White
Write-Host "`nNext: Run .\upload_lambda_via_s3.ps1 with the minimal package" -ForegroundColor Yellow
