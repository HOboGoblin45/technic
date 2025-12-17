"""
Hybrid API: Render + AWS Lambda
Routes cached scans to Render, heavy scans to Lambda
"""

import json
import time
import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
from technic_v4.scanner_core import ScanConfig
from technic_v4.cache.redis_cache import redis_cache
from technic_v4.infra.logging import get_logger

logger = get_logger()

# Initialize FastAPI
app = FastAPI(
    title="Technic Scanner API (Hybrid)",
    description="Hybrid architecture: Render for cached scans, Lambda for heavy lifting",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AWS Lambda client
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
LAMBDA_FUNCTION_NAME = os.environ.get('LAMBDA_FUNCTION_NAME', 'technic-scanner')
USE_LAMBDA = os.environ.get('USE_LAMBDA', 'true').lower() == 'true'

lambda_client = None
if USE_LAMBDA:
    try:
        lambda_client = boto3.client('lambda', region_name=AWS_REGION)
        logger.info(f"[HYBRID] Lambda client initialized for {LAMBDA_FUNCTION_NAME}")
    except Exception as e:
        logger.error(f"[HYBRID] Failed to initialize Lambda client: {e}")
        lambda_client = None


# Request/Response models
class ScanRequest(BaseModel):
    sectors: Optional[list] = None
    industries: Optional[list] = None
    max_symbols: int = 50
    min_tech_rating: float = 10.0
    min_price: float = 5.0
    min_dollar_vol: float = 1000000
    profile: str = "balanced"
    trade_style: str = "swing"
    lookback_days: int = 90
    account_size: float = 100000
    risk_pct: float = 2.0
    target_rr: float = 2.0
    force_lambda: bool = False  # Force Lambda execution for testing


class ScanResponse(BaseModel):
    cached: bool
    source: str  # "render_cache", "lambda", "render_compute"
    results: Dict[str, Any]
    execution_time: float
    lambda_info: Optional[Dict[str, Any]] = None


def get_cache_key(request: ScanRequest) -> str:
    """Generate cache key from scan request"""
    config_dict = request.dict()
    config_dict.pop('force_lambda', None)  # Don't include force_lambda in cache key
    sorted_config = json.dumps(config_dict, sort_keys=True)
    return f"hybrid_scan:{sorted_config}"


def check_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """Check if result is cached"""
    if not redis_cache.available:
        return None
    
    try:
        cached = redis_cache.get(cache_key)
        if cached:
            logger.info(f"[HYBRID] Cache hit: {cache_key[:50]}...")
            return json.loads(cached) if isinstance(cached, str) else cached
    except Exception as e:
        logger.error(f"[HYBRID] Cache read error: {e}")
    
    return None


def invoke_lambda(request: ScanRequest) -> Dict[str, Any]:
    """Invoke AWS Lambda for heavy computation"""
    if not lambda_client:
        raise HTTPException(
            status_code=503,
            detail="Lambda client not available. Check AWS credentials."
        )
    
    try:
        logger.info(f"[HYBRID] Invoking Lambda: {LAMBDA_FUNCTION_NAME}")
        
        # Prepare payload
        payload = request.dict()
        payload.pop('force_lambda', None)
        
        # Invoke Lambda
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION_NAME,
            InvocationType='RequestResponse',  # Synchronous
            Payload=json.dumps(payload)
        )
        
        # Parse response
        response_payload = json.loads(response['Payload'].read())
        
        if response_payload.get('statusCode') != 200:
            error_body = json.loads(response_payload.get('body', '{}'))
            raise HTTPException(
                status_code=response_payload.get('statusCode', 500),
                detail=error_body.get('error', 'Lambda execution failed')
            )
        
        # Parse successful response
        body = json.loads(response_payload['body'])
        logger.info(f"[HYBRID] Lambda completed in {body.get('execution_time', 0):.2f}s")
        
        return body
        
    except Exception as e:
        logger.error(f"[HYBRID] Lambda invocation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Lambda invocation failed: {str(e)}"
        )


def run_local_scan(request: ScanRequest) -> Dict[str, Any]:
    """Run scan locally on Render (fallback)"""
    from technic_v4.scanner_core import run_scan
    
    logger.info("[HYBRID] Running scan locally on Render")
    
    start_time = time.time()
    
    # Create config
    config = ScanConfig(
        sectors=request.sectors,
        industries=request.industries,
        max_symbols=request.max_symbols,
        min_tech_rating=request.min_tech_rating,
        min_price=request.min_price,
        min_dollar_vol=request.min_dollar_vol,
        profile=request.profile,
        trade_style=request.trade_style,
        lookback_days=request.lookback_days,
        account_size=request.account_size,
        risk_pct=request.risk_pct,
        target_rr=request.target_rr
    )
    
    # Run scan
    results_df, status_text, metrics = run_scan(config)
    
    execution_time = time.time() - start_time
    
    return {
        'cached': False,
        'source': 'render_compute',
        'results': {
            'symbols': results_df.to_dict('records') if not results_df.empty else [],
            'status': status_text,
            'metrics': metrics
        },
        'execution_time': execution_time
    }


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Technic Scanner API (Hybrid)",
        "version": "2.0.0",
        "architecture": "Render + AWS Lambda",
        "lambda_enabled": USE_LAMBDA and lambda_client is not None,
        "redis_available": redis_cache.available
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "lambda_available": lambda_client is not None,
        "redis_available": redis_cache.available,
        "lambda_function": LAMBDA_FUNCTION_NAME if lambda_client else None
    }


@app.post("/scan", response_model=ScanResponse)
async def scan(request: ScanRequest):
    """
    Main scan endpoint with hybrid architecture
    
    Flow:
    1. Check Redis cache
    2. If cached → return immediately
    3. If not cached:
       a. If Lambda available → use Lambda
       b. If Lambda not available → use Render
    4. Cache result
    """
    
    start_time = time.time()
    
    try:
        # Generate cache key
        cache_key = get_cache_key(request)
        
        # Check cache (unless force_lambda is true)
        if not request.force_lambda:
            cached_result = check_cache(cache_key)
            if cached_result:
                return ScanResponse(
                    cached=True,
                    source="render_cache",
                    results=cached_result,
                    execution_time=time.time() - start_time
                )
        
        # Not cached - decide where to compute
        if USE_LAMBDA and lambda_client and not request.force_lambda:
            # Use Lambda for heavy lifting
            logger.info("[HYBRID] Routing to Lambda")
            result = invoke_lambda(request)
            
            # Cache the result
            if redis_cache.available:
                try:
                    redis_cache.set(cache_key, json.dumps(result['results']), ttl=300)
                except Exception as e:
                    logger.error(f"[HYBRID] Cache write error: {e}")
            
            return ScanResponse(**result)
        
        else:
            # Use Render for computation (fallback)
            logger.info("[HYBRID] Routing to Render (Lambda not available)")
            result = run_local_scan(request)
            
            # Cache the result
            if redis_cache.available:
                try:
                    redis_cache.set(cache_key, json.dumps(result['results']), ttl=300)
                except Exception as e:
                    logger.error(f"[HYBRID] Cache write error: {e}")
            
            return ScanResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[HYBRID] Scan error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Scan failed: {str(e)}"
        )


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    if not redis_cache.available:
        return {"available": False}
    
    try:
        stats = redis_cache.get_stats()
        return stats
    except Exception as e:
        logger.error(f"[HYBRID] Cache stats error: {e}")
        return {"error": str(e)}


@app.post("/cache/clear")
async def clear_cache(pattern: str = "hybrid_scan:*"):
    """Clear cache by pattern"""
    if not redis_cache.available:
        raise HTTPException(status_code=503, detail="Cache not available")
    
    try:
        count = redis_cache.clear_pattern(pattern)
        return {"cleared": count, "pattern": pattern}
    except Exception as e:
        logger.error(f"[HYBRID] Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/lambda/info")
async def lambda_info():
    """Get Lambda configuration info"""
    return {
        "enabled": USE_LAMBDA,
        "available": lambda_client is not None,
        "function_name": LAMBDA_FUNCTION_NAME if lambda_client else None,
        "region": AWS_REGION
    }


@app.post("/lambda/test")
async def test_lambda(request: ScanRequest):
    """Test Lambda invocation directly"""
    if not lambda_client:
        raise HTTPException(
            status_code=503,
            detail="Lambda client not available"
        )
    
    try:
        result = invoke_lambda(request)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"[HYBRID] Starting API on port {port}")
    logger.info(f"[HYBRID] Lambda enabled: {USE_LAMBDA}")
    logger.info(f"[HYBRID] Lambda function: {LAMBDA_FUNCTION_NAME}")
    logger.info(f"[HYBRID] Redis available: {redis_cache.available}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
