# Deploy Lambda with Layers (Solves 250MB Limit)
# Splits dependencies into layers to bypass size restrictions

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Lambda Deployment with Layers" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$FUNCTION_NAME = "technic-scanner"
$REGION = "us-east-1"

# Step 1: Create NumPy/SciPy Layer
Write-Host "`n[1/6] Creating NumPy/SciPy layer..." -ForegroundColor Yellow
if (Test-Path "layer_numpy") { Remove-Item -Recurse -Force layer_numpy }
New-Item -ItemType Directory -Force -Path layer_numpy/python | Out-Null

pip install --target layer_numpy/python `
    --platform manylinux2014_x86_64 `
    --implementation cp `
    --python-version 3.11 `
    --only-binary=:all: `
    --no-cache-dir `
    numpy==1.24.3 `
    scipy==1.11.3 2>&1 | Out-Null

# Remove unnecessary files
Get-ChildItem -Path layer_numpy -Recurse -Include "*.dist-info","*.egg-info","__pycache__","*.pyc","*.md","*.txt" -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

if (Test-Path "numpy-scipy-layer.zip") { Remove-Item numpy-scipy-layer.zip -Force }
Compress-Archive -Path layer_numpy/* -DestinationPath numpy-scipy-layer.zip -CompressionLevel Optimal

Write-Host "  Layer 1 created: numpy-scipy-layer.zip" -ForegroundColor Green

# Step 2: Create Pandas/scikit-learn Layer  
Write-Host "`n[2/6] Creating Pandas/scikit-learn layer..." -ForegroundColor Yellow
if (Test-Path "layer_ml") { Remove-Item -Recurse -Force layer_ml }
New-Item -ItemType Directory -Force -Path layer_ml/python | Out-Null

pip install --target layer_ml/python `
    --platform manylinux2014_x86_64 `
    --implementation cp `
    --python-version 3.11 `
    --only-binary=:all: `
    --no-cache-dir `
    pandas==2.0.3 `
    scikit-learn==1.3.0 2>&1 | Out-Null

Get-ChildItem -Path layer_ml -Recurse -Include "*.dist-info","*.egg-info","__pycache__","*.pyc","*.md","*.txt" -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

if (Test-Path "pandas-sklearn-layer.zip") { Remove-Item pandas-sklearn-layer.zip -Force }
Compress-Archive -Path layer_ml/* -DestinationPath pandas-sklearn-layer.zip -CompressionLevel Optimal

Write-Host "  Layer 2 created: pandas-sklearn-layer.zip" -ForegroundColor Green

# Step 3: Create main package (code + Redis + small deps)
Write-Host "`n[3/6] Creating main Lambda package..." -ForegroundColor Yellow
if (Test-Path "lambda_main") { Remove-Item -Recurse -Force lambda_main }
New-Item -ItemType Directory -Force -Path lambda_main | Out-Null

Copy-Item lambda_deploy/lambda_function.py lambda_main/
Copy-Item -Recurse lambda_deploy/technic_v4 lambda_main/

pip install --target lambda_main `
    --platform manylinux2014_x86_64 `
    --implementation cp `
    --python-version 3.11 `
    --only-binary=:all: `
    --no-cache-dir `
    redis==5.0.0 `
    requests==2.31.0 `
    polygon-api-client==1.12.5 2>&1 | Out-Null

Get-ChildItem -Path lambda_main -Recurse -Include "*.dist-info","*.egg-info","__pycache__","*.pyc" -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

if (Test-Path "lambda-main.zip") { Remove-Item lambda-main.zip -Force }
Compress-Archive -Path lambda_main/* -DestinationPath lambda-main.zip -CompressionLevel Optimal

$mainSize = (Get-Item lambda-main.zip).Length / 1MB
Write-Host "  Main package created: lambda-main.zip ($([math]::Round($mainSize, 2)) MB)" -ForegroundColor Green

# Step 4: Upload layers to S3
Write-Host "`n[4/6] Uploading layers to S3..." -ForegroundColor Yellow
$BUCKET = "technic-lambda-deploy-9737"

aws s3 cp numpy-scipy-layer.zip s3://$BUCKET/layers/ 2>&1 | Out-Null
aws s3 cp pandas-sklearn-layer.zip s3://$BUCKET/layers/ 2>&1 | Out-Null
Write-Host "  Layers uploaded to S3" -ForegroundColor Green

# Step 5: Publish layers
Write-Host "`n[5/6] Publishing Lambda layers..." -ForegroundColor Yellow

$layer1 = aws lambda publish-layer-version `
    --layer-name numpy-scipy-layer `
    --description "NumPy 1.24.3 + SciPy 1.11.3" `
    --content S3Bucket=$BUCKET,S3Key=layers/numpy-scipy-layer.zip `
    --compatible-runtimes python3.11 `
    --region $REGION | ConvertFrom-Json

$layer2 = aws lambda publish-layer-version `
    --layer-name pandas-sklearn-layer `
    --description "Pandas 2.0.3 + scikit-learn 1.3.0" `
    --content S3Bucket=$BUCKET,S3Key=layers/pandas-sklearn-layer.zip `
    --compatible-runtimes python3.11 `
    --region $REGION | ConvertFrom-Json

Write-Host "  Layer 1 ARN: $($layer1.LayerVersionArn)" -ForegroundColor Cyan
Write-Host "  Layer 2 ARN: $($layer2.LayerVersionArn)" -ForegroundColor Cyan

# Step 6: Update Lambda function
Write-Host "`n[6/6] Updating Lambda function..." -ForegroundColor Yellow

# Upload main package
aws s3 cp lambda-main.zip s3://$BUCKET/ 2>&1 | Out-Null

# Update function code
aws lambda update-function-code `
    --function-name $FUNCTION_NAME `
    --s3-bucket $BUCKET `
    --s3-key lambda-main.zip `
    --region $REGION 2>&1 | Out-Null

Start-Sleep -Seconds 5

# Attach layers
aws lambda update-function-configuration `
    --function-name $FUNCTION_NAME `
    --layers $layer1.LayerVersionArn $layer2.LayerVersionArn `
    --region $REGION 2>&1 | Out-Null

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Function: $FUNCTION_NAME" -ForegroundColor Cyan
Write-Host "Main package: $([math]::Round($mainSize, 2)) MB (under 50MB limit!)" -ForegroundColor Cyan
Write-Host "Layer 1: NumPy + SciPy" -ForegroundColor Cyan
Write-Host "Layer 2: Pandas + scikit-learn" -ForegroundColor Cyan
Write-Host "`nNext: Test the function with .\test_lambda.ps1" -ForegroundColor Yellow
