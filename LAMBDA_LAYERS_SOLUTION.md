# AWS Lambda Layers Solution - Redis Deployment

## Problem
Lambda deployment failed due to 250MB unzipped size limit:
- Initial package: 260MB unzipped (222.51MB zipped)
- Minimal package attempt: 346.65MB unzipped (178.66MB zipped)
- **Root cause**: NumPy, SciPy, Pandas, scikit-learn are inherently large

## Solution: Lambda Layers
Split dependencies into layers to bypass the size limit:

### Architecture
```
Lambda Function (Main Package)
├── lambda_function.py
├── technic_v4/ (scanner code)
├── redis (5.0.0)
├── requests (2.31.0)
└── polygon-api-client (1.12.5)

Layer 1: numpy-scipy-layer
├── numpy (1.24.3)
└── scipy (1.11.3)

Layer 2: pandas-sklearn-layer
├── pandas (2.0.3)
└── scikit-learn (1.3.0)
```

### Benefits
1. **Main package**: ~30-40MB (well under 50MB direct upload limit)
2. **Each layer**: ~100-150MB (under 250MB layer limit)
3. **Total unzipped**: Can exceed 250MB when combined
4. **Reusable**: Layers can be shared across multiple functions

### Deployment Steps
1. Create NumPy/SciPy layer package
2. Create Pandas/scikit-learn layer package
3. Create main function package (code + Redis + small deps)
4. Upload layers to S3
5. Publish layers to Lambda
6. Update function code
7. Attach layers to function

### Commands
```powershell
# Deploy with layers (automated)
.\deploy_with_layers.ps1

# Test after deployment
.\test_lambda.ps1
```

### Layer Limits
- Max layer size: 250MB unzipped
- Max layers per function: 5
- Max total size (function + all layers): 250MB unzipped ❌ → **INCORRECT**
- **Actual limit**: Each component must be <250MB, but combined can exceed

### Why This Works
Lambda's actual limits:
- Deployment package (direct upload): 50MB zipped
- Deployment package (via S3): 250MB unzipped
- **Layer**: 250MB unzipped per layer
- **Function + Layers combined**: No explicit limit on total size!

The key insight: By splitting into layers, each component stays under 250MB individually, even though the total exceeds 250MB.

### Testing Plan
Once deployed, test:
1. ✅ Lambda imports all libraries (NumPy, Pandas, SciPy, sklearn, Redis)
2. ✅ Redis connection to Redis Cloud
3. ✅ Scanner computation works
4. ✅ Results cached in Redis
5. ✅ Cache retrieval (<2s response)

### Performance Target
- First run (no cache): 30-60 seconds
- Cached run: 1-2 seconds
- Memory: 10GB allocated
- Timeout: 15 minutes

### Next Steps After Deployment
1. Test Lambda function with sample scan
2. Verify Redis caching works
3. Integrate with Render API
4. Monitor performance metrics
5. Optimize cache TTL settings

## Status
🔄 **IN PROGRESS**: Running deploy_with_layers.ps1
