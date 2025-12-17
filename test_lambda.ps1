# Test Lambda Function
# This script invokes the Lambda function and displays the results

$ErrorActionPreference = "Stop"
$awsCmd = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing Lambda Function" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$functionName = "technic-scanner"
$region = "us-east-1"

# Create test payload as JSON string
$payload = @'
{
    "action": "scan",
    "config": {
        "max_symbols": 5,
        "sectors": ["Technology"],
        "min_tech_rating": 10.0
    }
}
'@

# Save to temp file with UTF-8 encoding (no BOM)
$tempFile = [System.IO.Path]::GetTempFileName()
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($tempFile, $payload, $utf8NoBom)

Write-Host "[1/3] Invoking Lambda function..." -ForegroundColor Yellow
Write-Host "  Function: $functionName" -ForegroundColor Gray
Write-Host "  Region: $region" -ForegroundColor Gray
Write-Host "  Test: Scanning 5 Technology stocks" -ForegroundColor Gray
Write-Host ""

try {
    $result = & $awsCmd lambda invoke `
        --function-name $functionName `
        --payload "file://$tempFile" `
        --region $region `
        --cli-binary-format raw-in-base64-out `
        response.json 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK Lambda invoked successfully" -ForegroundColor Green
    } else {
        Write-Host "  ERROR Lambda invocation failed" -ForegroundColor Red
        Write-Host "  Error: $result" -ForegroundColor Red
        Remove-Item $tempFile -ErrorAction SilentlyContinue
        exit 1
    }
} catch {
    Write-Host "  ERROR invoking Lambda: $_" -ForegroundColor Red
    Remove-Item $tempFile -ErrorAction SilentlyContinue
    exit 1
}

# Clean up temp file
Remove-Item $tempFile -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "[2/3] Reading response..." -ForegroundColor Yellow

if (Test-Path "response.json") {
    try {
        $response = Get-Content "response.json" -Raw | ConvertFrom-Json
        
        Write-Host ""
        Write-Host "  Response Summary:" -ForegroundColor White
        Write-Host "  ================" -ForegroundColor White
        
        if ($response.statusCode) {
            Write-Host "  Status Code: $($response.statusCode)" -ForegroundColor Gray
        }
        
        if ($response.body) {
            $body = $response.body | ConvertFrom-Json
            
            if ($body.results) {
                Write-Host "  Results Found: $($body.results.Count)" -ForegroundColor Green
                Write-Host ""
                Write-Host "  Top Results:" -ForegroundColor White
                $body.results | Select-Object -First 5 | ForEach-Object {
                    Write-Host "    - $($_.Symbol): Score $($_.TechRating)" -ForegroundColor Gray
                }
            }
            
            if ($body.performance_metrics) {
                Write-Host ""
                Write-Host "  Performance:" -ForegroundColor White
                Write-Host "    Duration: $($body.performance_metrics.total_seconds)s" -ForegroundColor Gray
                Write-Host "    Symbols Scanned: $($body.performance_metrics.symbols_scanned)" -ForegroundColor Gray
                Write-Host "    Speed: $($body.performance_metrics.symbols_per_second) sym/s" -ForegroundColor Gray
            }
            
            if ($body.error) {
                Write-Host ""
                Write-Host "  Error: $($body.error)" -ForegroundColor Red
                if ($body.details) {
                    Write-Host "  Details: $($body.details)" -ForegroundColor Yellow
                }
            }
        }
        
        Write-Host ""
        Write-Host "  Full response saved to: response.json" -ForegroundColor Gray
        
    } catch {
        Write-Host "  ERROR parsing response: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "  Raw response:" -ForegroundColor Yellow
        Get-Content "response.json"
    }
} else {
    Write-Host "  ERROR Response file not found" -ForegroundColor Red
}

Write-Host ""
Write-Host "[3/3] Checking CloudWatch logs..." -ForegroundColor Yellow
$logsUrl = "https://console.aws.amazon.com/cloudwatch/home?region=$region#logsV2:log-groups/log-group/`$252Faws`$252Flambda`$252F$functionName"
Write-Host "  View logs at:" -ForegroundColor White
Write-Host "  $logsUrl" -ForegroundColor Cyan

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
