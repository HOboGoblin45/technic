# Configure Lambda Environment Variables and Settings
# This script sets up all required environment variables for the Lambda function

$ErrorActionPreference = "Stop"
$awsCmd = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Lambda Configuration" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$functionName = "technic-scanner"
$region = "us-east-1"

# Step 1: Set environment variables
Write-Host "[1/3] Setting environment variables..." -ForegroundColor Yellow

# Get Polygon API key from user
Write-Host ""
Write-Host "Please enter your Polygon.io API key:" -ForegroundColor White
Write-Host "(You can find this in your Polygon.io dashboard)" -ForegroundColor Gray
$polygonKey = Read-Host "Polygon API Key"

if ([string]::IsNullOrWhiteSpace($polygonKey)) {
    Write-Host "ERROR: Polygon API key is required!" -ForegroundColor Red
    exit 1
}

# Optional: Redis URL
Write-Host ""
Write-Host "Do you have a Redis URL? (Y/N)" -ForegroundColor White
Write-Host "(Optional - improves performance with caching)" -ForegroundColor Gray
$hasRedis = Read-Host "  "
$redisUrl = ""
if ($hasRedis -eq 'Y' -or $hasRedis -eq 'y') {
    Write-Host "Enter Redis URL:" -ForegroundColor White
    $redisUrl = Read-Host "  "
}

# Build environment variables JSON
$envVars = @{
    "POLYGON_API_KEY" = $polygonKey
    "ENVIRONMENT" = "production"
    "LOG_LEVEL" = "INFO"
}

if (-not [string]::IsNullOrWhiteSpace($redisUrl)) {
    $envVars["REDIS_URL"] = $redisUrl
}

$envJson = $envVars | ConvertTo-Json -Compress

Write-Host "  Setting environment variables..." -ForegroundColor Gray
try {
    & $awsCmd lambda update-function-configuration `
        --function-name $functionName `
        --environment "Variables=$envJson" `
        --region $region `
        --output json | Out-Null
    Write-Host "  OK Environment variables set" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Failed to set environment variables: $_" -ForegroundColor Red
    exit 1
}

# Step 2: Verify configuration
Write-Host "[2/3] Verifying configuration..." -ForegroundColor Yellow
Start-Sleep -Seconds 2  # Wait for Lambda to update

try {
    $config = & $awsCmd lambda get-function-configuration `
        --function-name $functionName `
        --region $region `
        --output json | ConvertFrom-Json
    
    Write-Host ""
    Write-Host "  Current Configuration:" -ForegroundColor White
    Write-Host "  =====================" -ForegroundColor White
    Write-Host "  Function Name: $($config.FunctionName)" -ForegroundColor Gray
    Write-Host "  Runtime: $($config.Runtime)" -ForegroundColor Gray
    Write-Host "  Memory: $($config.MemorySize) MB" -ForegroundColor Gray
    Write-Host "  Timeout: $($config.Timeout) seconds" -ForegroundColor Gray
    Write-Host "  Code Size: $([math]::Round($config.CodeSize / 1MB, 2)) MB" -ForegroundColor Gray
    Write-Host "  Last Modified: $($config.LastModified)" -ForegroundColor Gray
    
    if ($config.Environment.Variables) {
        Write-Host ""
        Write-Host "  Environment Variables:" -ForegroundColor White
        $config.Environment.Variables.PSObject.Properties | ForEach-Object {
            if ($_.Name -eq "POLYGON_API_KEY") {
                Write-Host "    $($_.Name): ****$(($_.Value).Substring([Math]::Max(0, $_.Value.Length - 4)))" -ForegroundColor Gray
            } else {
                Write-Host "    $($_.Name): $($_.Value)" -ForegroundColor Gray
            }
        }
    }
    
    Write-Host ""
    Write-Host "  OK Configuration verified" -ForegroundColor Green
} catch {
    Write-Host "  WARNING: Could not verify configuration: $_" -ForegroundColor Yellow
}

# Step 3: Create test event
Write-Host "[3/3] Creating test event..." -ForegroundColor Yellow

$testEvent = @{
    "action" = "scan"
    "config" = @{
        "max_symbols" = 5
        "sectors" = @("Technology")
        "min_tech_rating" = 10.0
    }
} | ConvertTo-Json -Depth 10

$testEventFile = "lambda_test_event.json"
$testEvent | Out-File -FilePath $testEventFile -Encoding UTF8

Write-Host "  OK Test event created: $testEventFile" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Configuration Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Test the Lambda function:" -ForegroundColor White
Write-Host "   & '$awsCmd' lambda invoke ``" -ForegroundColor Gray
Write-Host "     --function-name $functionName ``" -ForegroundColor Gray
Write-Host "     --payload file://$testEventFile ``" -ForegroundColor Gray
Write-Host "     --region $region ``" -ForegroundColor Gray
Write-Host "     response.json" -ForegroundColor Gray
Write-Host ""
Write-Host "2. View the response:" -ForegroundColor White
Write-Host "   Get-Content response.json | ConvertFrom-Json | ConvertTo-Json -Depth 10" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Check CloudWatch logs:" -ForegroundColor White
Write-Host "   https://console.aws.amazon.com/cloudwatch/home?region=$region#logsV2:log-groups/log-group/`$252Faws`$252Flambda`$252F$functionName" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Integrate with your Render API (see HYBRID_DEPLOYMENT_GUIDE.md)" -ForegroundColor White
Write-Host ""
