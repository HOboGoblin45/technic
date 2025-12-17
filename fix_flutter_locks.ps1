# Flutter File Lock Fix Script
# Run this as Administrator

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Flutter File Lock Fix Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✓ Running as Administrator" -ForegroundColor Green
Write-Host ""

# Step 1: Kill all Flutter/Dart processes
Write-Host "Step 1: Killing all Flutter/Dart processes..." -ForegroundColor Yellow
try {
    taskkill /F /IM dart.exe /T 2>$null
    taskkill /F /IM flutter.exe /T 2>$null
    taskkill /F /IM chrome.exe /T 2>$null
    Write-Host "✓ Processes killed" -ForegroundColor Green
} catch {
    Write-Host "✓ No processes to kill" -ForegroundColor Green
}
Write-Host ""

# Wait a moment
Start-Sleep -Seconds 3

# Step 2: Delete build folders
Write-Host "Step 2: Deleting build folders..." -ForegroundColor Yellow
$projectPath = "C:\Users\ccres\OneDrive\Desktop\technic-clean\technic_mobile"

if (Test-Path $projectPath) {
    Set-Location $projectPath
    
    # Delete build folder
    if (Test-Path "build") {
        try {
            Remove-Item -Path "build" -Recurse -Force -ErrorAction Stop
            Write-Host "✓ Deleted build/" -ForegroundColor Green
        } catch {
            Write-Host "⚠ Could not delete build/ (may not exist)" -ForegroundColor Yellow
        }
    }
    
    # Delete .dart_tool folder
    if (Test-Path ".dart_tool") {
        try {
            Remove-Item -Path ".dart_tool" -Recurse -Force -ErrorAction Stop
            Write-Host "✓ Deleted .dart_tool/" -ForegroundColor Green
        } catch {
            Write-Host "⚠ Could not delete .dart_tool/ (may not exist)" -ForegroundColor Yellow
        }
    }
    
    # Delete ephemeral folder
    if (Test-Path "windows\flutter\ephemeral") {
        try {
            Remove-Item -Path "windows\flutter\ephemeral" -Recurse -Force -ErrorAction Stop
            Write-Host "✓ Deleted windows/flutter/ephemeral/" -ForegroundColor Green
        } catch {
            Write-Host "⚠ Could not delete ephemeral/ (may not exist)" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "✗ Project path not found: $projectPath" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Wait a moment
Start-Sleep -Seconds 3

# Step 3: Disable Windows Search temporarily
Write-Host "Step 3: Temporarily disabling Windows Search..." -ForegroundColor Yellow
try {
    Stop-Service -Name "WSearch" -Force -ErrorAction Stop
    Write-Host "✓ Windows Search stopped" -ForegroundColor Green
} catch {
    Write-Host "⚠ Could not stop Windows Search (may already be stopped)" -ForegroundColor Yellow
}
Write-Host ""

# Step 4: Add exclusions
Write-Host "Step 4: Adding Windows Defender exclusions..." -ForegroundColor Yellow
try {
    Add-MpPreference -ExclusionPath $projectPath -ErrorAction Stop
    Write-Host "✓ Added exclusion for $projectPath" -ForegroundColor Green
} catch {
    Write-Host "⚠ Could not add exclusion (may already exist)" -ForegroundColor Yellow
}
Write-Host ""

# Step 5: Instructions for user
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Close this Administrator PowerShell window" -ForegroundColor White
Write-Host "2. Open a NEW regular PowerShell (not as admin)" -ForegroundColor White
Write-Host "3. Run these commands:" -ForegroundColor White
Write-Host ""
Write-Host "   cd C:\Users\ccres\OneDrive\Desktop\technic-clean\technic_mobile" -ForegroundColor Cyan
Write-Host "   flutter run -d chrome" -ForegroundColor Cyan
Write-Host ""
Write-Host "4. After Flutter starts, run this to re-enable Windows Search:" -ForegroundColor White
Write-Host ""
Write-Host "   Start-Service -Name 'WSearch'" -ForegroundColor Cyan
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Option to re-enable Windows Search now
$response = Read-Host "Do you want to re-enable Windows Search now? (y/n)"
if ($response -eq 'y' -or $response -eq 'Y') {
    try {
        Start-Service -Name "WSearch"
        Write-Host "✓ Windows Search re-enabled" -ForegroundColor Green
    } catch {
        Write-Host "⚠ Could not start Windows Search" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Press Enter to exit..." -ForegroundColor Yellow
Read-Host
