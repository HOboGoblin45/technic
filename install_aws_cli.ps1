# Install AWS CLI for Windows
# This script downloads and installs AWS CLI v2

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AWS CLI Installation Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if AWS CLI is already installed
$awsInstalled = Get-Command aws -ErrorAction SilentlyContinue

if ($awsInstalled) {
    Write-Host "AWS CLI is already installed!" -ForegroundColor Green
    Write-Host "Version: " -NoNewline
    aws --version
    Write-Host ""
    Write-Host "If you want to reinstall, uninstall first from Control Panel." -ForegroundColor Yellow
    exit 0
}

Write-Host "AWS CLI not found. Installing..." -ForegroundColor Yellow
Write-Host ""

# Download AWS CLI installer
$installerUrl = "https://awscli.amazonaws.com/AWSCLIV2.msi"
$installerPath = "$env:TEMP\AWSCLIV2.msi"

Write-Host "Downloading AWS CLI installer..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath
    Write-Host "✓ Download complete" -ForegroundColor Green
} catch {
    Write-Host "✗ Download failed: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please download manually from:" -ForegroundColor Yellow
    Write-Host "  https://awscli.amazonaws.com/AWSCLIV2.msi" -ForegroundColor White
    exit 1
}

# Install AWS CLI
Write-Host ""
Write-Host "Installing AWS CLI..." -ForegroundColor Yellow
Write-Host "  (This may take a few minutes and require administrator approval)" -ForegroundColor Gray

try {
    Start-Process msiexec.exe -ArgumentList "/i `"$installerPath`" /quiet /norestart" -Wait -NoNewWindow
    Write-Host "✓ Installation complete" -ForegroundColor Green
} catch {
    Write-Host "✗ Installation failed: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install manually:" -ForegroundColor Yellow
    Write-Host "  1. Double-click: $installerPath" -ForegroundColor White
    Write-Host "  2. Follow the installation wizard" -ForegroundColor White
    exit 1
}

# Clean up
Remove-Item $installerPath -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "IMPORTANT: You must restart PowerShell for changes to take effect!" -ForegroundColor Yellow
Write-Host ""
Write-Host "After restarting PowerShell:" -ForegroundColor Cyan
Write-Host "  1. Verify installation: aws --version" -ForegroundColor White
Write-Host "  2. Configure credentials: aws configure" -ForegroundColor White
Write-Host "  3. Upload Lambda: aws lambda update-function-code --function-name technic-scanner --zip-file fileb://technic-scanner.zip" -ForegroundColor White
Write-Host ""
