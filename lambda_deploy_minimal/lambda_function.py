"""
AWS Lambda Scanner Function
Handles heavy scanner workloads with 10GB memory and 6 vCPUs
"""

import json
import os
import time
from typing import Dict, Any, Optional
import redis
from technic_v4.scanner_core import run_scan, ScanConfig
from technic_v4.infra.logging import get_logger

logger = get_logger()

# Initialize Redis connection (shared with Render)
REDIS_URL = os.environ.get('REDIS_URL')
redis_client = None

if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        logger.info("[LAMBDA] Connected to Redis Cloud")
    except Exception as e:
        logger.error(f"[LAMBDA] Failed to connect to Redis: {e}")
        redis_client = None
else:
    logger.warning("[LAMBDA] No REDIS_URL provided")


def get_cache_key(config: Dict[str, Any]) -> str:
    """Generate cache key from scan configuration"""
    # Sort keys for consistent cache keys
    sorted_config = json.dumps(config, sort_keys=True)
    return f"lambda_scan:{sorted_config}"


def get_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    """Check if result is cached in Redis"""
    if not redis_client:
        return None
    
    try:
        cached = redis_client.get(cache_key)
        if cached:
            logger.info(f"[LAMBDA] Cache hit for key: {cache_key[:50]}...")
            return json.loads(cached)
    except Exception as e:
        logger.error(f"[LAMBDA] Cache read error: {e}")
    
    return None


def cache_result(cache_key: str, result: Dict[str, Any], ttl: int = 300):
    """Cache result in Redis with TTL"""
    if not redis_client:
        return
    
    try:
        redis_client.setex(
            cache_key,
            ttl,  # 5 minutes default
            json.dumps(result)
        )
        logger.info(f"[LAMBDA] Cached result for {ttl}s: {cache_key[:50]}...")
    except Exception as e:
        logger.error(f"[LAMBDA] Cache write error: {e}")


def lambda_handler(event, context):
    """
    AWS Lambda handler for scanner
    
    Event format:
    {
        "sectors": ["Technology", "Healthcare"],
        "industries": ["Software", "Biotechnology"],
        "max_symbols": 50,
        "min_tech_rating": 10.0,
        "min_price": 5.0,
        "min_dollar_vol": 1000000,
        "profile": "aggressive",
        "trade_style": "swing",
        "lookback_days": 90,
        "account_size": 100000,
        "risk_pct": 2.0,
        "target_rr": 2.0
    }
    
    Returns:
    {
        "statusCode": 200,
        "body": {
            "cached": false,
            "source": "lambda",
            "results": {
                "symbols": [...],
                "status": "...",
                "metrics": {...}
            },
            "execution_time": 45.2,
            "lambda_info": {
                "memory_limit": 10240,
                "memory_used": 8500,
                "time_remaining": 850000
            }
        }
    }
    """
    
    start_time = time.time()
    
    try:
        # Log Lambda context
        logger.info(f"[LAMBDA] Function: {context.function_name}")
        logger.info(f"[LAMBDA] Memory: {context.memory_limit_in_mb}MB")
        logger.info(f"[LAMBDA] Request ID: {context.request_id}")
        
        # Parse event
        if isinstance(event, str):
            event = json.loads(event)
        
        # Check cache first
        cache_key = get_cache_key(event)
        cached_result = get_cached_result(cache_key)
        
        if cached_result:
            execution_time = time.time() - start_time
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'cached': True,
                    'source': 'redis',
                    'results': cached_result,
                    'execution_time': execution_time,
                    'lambda_info': {
                        'memory_limit': context.memory_limit_in_mb,
                        'time_remaining': context.get_remaining_time_in_millis()
                    }
                })
            }
        
        # Create scan configuration
        config = ScanConfig(
            sectors=event.get('sectors'),
            industries=event.get('industries'),
            max_symbols=event.get('max_symbols', 50),
            min_tech_rating=event.get('min_tech_rating', 10.0),
            min_price=event.get('min_price', 5.0),
            min_dollar_vol=event.get('min_dollar_vol', 1000000),
            profile=event.get('profile', 'balanced'),
            trade_style=event.get('trade_style', 'swing'),
            lookback_days=event.get('lookback_days', 90),
            account_size=event.get('account_size', 100000),
            risk_pct=event.get('risk_pct', 2.0),
            target_rr=event.get('target_rr', 2.0)
        )
        
        logger.info(f"[LAMBDA] Starting scan with config: {config}")
        
        # Run scan
        scan_start = time.time()
        results_df, status_text, metrics = run_scan(config)
        scan_time = time.time() - scan_start
        
        logger.info(f"[LAMBDA] Scan completed in {scan_time:.2f}s")
        logger.info(f"[LAMBDA] Found {len(results_df)} results")
        
        # Convert DataFrame to dict
        results = {
            'symbols': results_df.to_dict('records') if not results_df.empty else [],
            'status': status_text,
            'metrics': metrics,
            'scan_time': scan_time
        }
        
        # Cache results
        cache_result(cache_key, results, ttl=300)  # 5 minutes
        
        execution_time = time.time() - start_time
        
        # Return response
        return {
            'statusCode': 200,
            'body': json.dumps({
                'cached': False,
                'source': 'lambda',
                'results': results,
                'execution_time': execution_time,
                'lambda_info': {
                    'memory_limit': context.memory_limit_in_mb,
                    'memory_used': metrics.get('memory_used_mb', 0),
                    'time_remaining': context.get_remaining_time_in_millis()
                }
            })
        }
        
    except Exception as e:
        logger.error(f"[LAMBDA] Error: {e}", exc_info=True)
        
        execution_time = time.time() - start_time
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'error_type': type(e).__name__,
                'execution_time': execution_time,
                'lambda_info': {
                    'memory_limit': context.memory_limit_in_mb,
                    'time_remaining': context.get_remaining_time_in_millis()
                }
            })
        }


# For local testing
if __name__ == "__main__":
    # Mock Lambda context
    class MockContext:
        function_name = "technic-scanner-local"
        memory_limit_in_mb = 10240
        request_id = "local-test"
        
        def get_remaining_time_in_millis(self):
            return 900000  # 15 minutes
    
    # Test event
    test_event = {
        "sectors": ["Technology"],
        "max_symbols": 10,
        "min_tech_rating": 10.0,
        "profile": "aggressive"
    }
    
    # Run handler
    result = lambda_handler(test_event, MockContext())
    
    # Print result
    print(json.dumps(json.loads(result['body']), indent=2))
