# Phase 3D-B: Enhanced Error Messages & Loading States

## Overview

Building on Phase 3D-A (Progress Callbacks, Cache Status, Performance Metrics), we'll now add:
1. Enhanced error messages with actionable suggestions
2. Loading states with real-time progress
3. Error categorization and recovery strategies

## Goals

- **Better UX**: Users understand what went wrong and how to fix it
- **Transparency**: Show real-time progress during scans
- **Reliability**: Graceful error handling with recovery suggestions

## Implementation Tasks

### Task 1: Enhanced Error Messages (1-2 days)

#### 1.1 Error Classification System
Create error categories with user-friendly messages:

```python
class ScanError:
    API_ERROR = "api_error"           # Polygon API issues
    CACHE_ERROR = "cache_error"       # Redis connection issues
    DATA_ERROR = "data_error"         # Missing/invalid data
    TIMEOUT_ERROR = "timeout_error"   # Operation timeout
    CONFIG_ERROR = "config_error"     # Invalid configuration
    SYSTEM_ERROR = "system_error"     # Unexpected errors
```

#### 1.2 Error Message Structure
```python
{
    "error_type": "api_error",
    "message": "Unable to fetch market data",
    "details": "Polygon API rate limit exceeded",
    "suggestion": "Please wait 60 seconds and try again",
    "retry_after": 60,
    "recoverable": True,
    "affected_symbols": ["AAPL", "MSFT"]
}
```

#### 1.3 Files to Modify
- `technic_v4/scanner_core.py` - Add error handling wrapper
- `technic_v4/data_engine.py` - Enhance API error messages
- `technic_v4/cache/redis_cache.py` - Better cache error messages
- Create: `technic_v4/errors.py` - Error classification module

### Task 2: Loading States with Progress (1 day)

#### 2.1 Progress State Structure
```python
{
    "stage": "scanning",
    "current": 45,
    "total": 100,
    "message": "Analyzing AAPL...",
    "progress_pct": 45.0,
    "metadata": {
        "symbol": "AAPL",
        "sector": "Technology",
        "elapsed_time": 12.5,
        "estimated_remaining": 15.2,
        "symbols_per_second": 3.6
    }
}
```

#### 2.2 Integration Points
- Use existing progress callbacks from Phase 3D-A Task 1
- Add time estimation based on current throughput
- Include cache hit rate in progress updates
- Show which stage is running (prefetch, scan, filter, etc.)

#### 2.3 Files to Modify
- `technic_v4/scanner_core.py` - Enhance progress callbacks
- Add: `technic_v4/progress.py` - Progress tracking utilities

### Task 3: Error Recovery Strategies (1 day)

#### 3.1 Automatic Retry Logic
```python
@retry(
    max_attempts=3,
    backoff=exponential,
    exceptions=[APIError, TimeoutError]
)
def fetch_data(symbol):
    # Fetch logic with automatic retry
    pass
```

#### 3.2 Graceful Degradation
- If API fails: Use cached data (if available)
- If cache fails: Continue without cache
- If symbol fails: Skip and continue with others
- If critical error: Return partial results

#### 3.3 Files to Modify
- `technic_v4/scanner_core.py` - Add retry decorators
- `technic_v4/data_engine.py` - Implement fallback strategies
- Create: `technic_v4/retry.py` - Retry utilities

## Detailed Implementation

### Step 1: Create Error Classification Module

**File**: `technic_v4/errors.py`

```python
"""
Enhanced error handling for Technic Scanner
"""
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ErrorType(Enum):
    """Error categories for better user communication"""
    API_ERROR = "api_error"
    CACHE_ERROR = "cache_error"
    DATA_ERROR = "data_error"
    TIMEOUT_ERROR = "timeout_error"
    CONFIG_ERROR = "config_error"
    SYSTEM_ERROR = "system_error"


@dataclass
class ScanError:
    """Structured error information"""
    error_type: ErrorType
    message: str
    details: str
    suggestion: str
    recoverable: bool = True
    retry_after: Optional[int] = None
    affected_symbols: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses"""
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "details": self.details,
            "suggestion": self.suggestion,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "affected_symbols": self.affected_symbols,
            "metadata": self.metadata or {}
        }


# Predefined error messages
ERROR_MESSAGES = {
    ErrorType.API_ERROR: {
        "rate_limit": ScanError(
            error_type=ErrorType.API_ERROR,
            message="API rate limit exceeded",
            details="Too many requests to Polygon API",
            suggestion="Please wait 60 seconds and try again, or upgrade your API plan",
            retry_after=60
        ),
        "connection": ScanError(
            error_type=ErrorType.API_ERROR,
            message="Unable to connect to market data provider",
            details="Network connection to Polygon API failed",
            suggestion="Check your internet connection and try again"
        ),
        "invalid_key": ScanError(
            error_type=ErrorType.API_ERROR,
            message="Invalid API key",
            details="Polygon API key is invalid or expired",
            suggestion="Check your POLYGON_API_KEY environment variable",
            recoverable=False
        )
    },
    ErrorType.CACHE_ERROR: {
        "connection": ScanError(
            error_type=ErrorType.CACHE_ERROR,
            message="Cache unavailable",
            details="Unable to connect to Redis cache",
            suggestion="Continuing without cache (slower performance). Check Redis connection."
        ),
        "timeout": ScanError(
            error_type=ErrorType.CACHE_ERROR,
            message="Cache operation timeout",
            details="Redis operation took too long",
            suggestion="Cache will be bypassed for this scan"
        )
    },
    ErrorType.DATA_ERROR: {
        "missing": ScanError(
            error_type=ErrorType.DATA_ERROR,
            message="Insufficient data",
            details="Not enough historical data available for analysis",
            suggestion="Symbol will be skipped. Try increasing lookback period."
        ),
        "invalid": ScanError(
            error_type=ErrorType.DATA_ERROR,
            message="Invalid data format",
            details="Market data format is unexpected",
            suggestion="Symbol will be skipped. This may indicate a data provider issue."
        )
    },
    ErrorType.TIMEOUT_ERROR: {
        "scan": ScanError(
            error_type=ErrorType.TIMEOUT_ERROR,
            message="Scan timeout",
            details="Scan took longer than expected",
            suggestion="Try reducing the number of symbols or increasing timeout limit"
        )
    },
    ErrorType.CONFIG_ERROR: {
        "invalid": ScanError(
            error_type=ErrorType.CONFIG_ERROR,
            message="Invalid configuration",
            details="Scan configuration contains invalid parameters",
            suggestion="Check your scan settings and try again",
            recoverable=False
        )
    }
}


def get_error_message(error_type: ErrorType, error_key: str, **kwargs) -> ScanError:
    """
    Get a predefined error message with optional customization
    
    Args:
        error_type: Type of error
        error_key: Specific error within the type
        **kwargs: Additional fields to override
    
    Returns:
        ScanError instance
    """
    error = ERROR_MESSAGES.get(error_type, {}).get(error_key)
    if not error:
        # Fallback to generic error
        error = ScanError(
            error_type=ErrorType.SYSTEM_ERROR,
            message="An unexpected error occurred",
            details=str(kwargs.get("details", "Unknown error")),
            suggestion="Please try again or contact support"
        )
    
    # Override with provided kwargs
    if kwargs:
        error_dict = error.to_dict()
        error_dict.update(kwargs)
        error = ScanError(**error_dict)
    
    return error
```

### Step 2: Enhance Scanner Error Handling

**File**: `technic_v4/scanner_core.py` (additions)

```python
from technic_v4.errors import ScanError, ErrorType, get_error_message

def run_scan(
    config: Optional["ScanConfig"] = None,
    progress_cb: Optional[ProgressCallback] = None,
    error_cb: Optional[Callable[[ScanError], None]] = None  # NEW: Error callback
) -> Tuple[pd.DataFrame, str, dict]:
    """
    Run scanner with enhanced error handling
    
    Args:
        config: Scan configuration
        progress_cb: Progress callback
        error_cb: Error callback for reporting issues
    
    Returns:
        (results_df, status_message, performance_metrics)
    """
    errors_encountered = []
    
    try:
        # Existing scan logic...
        pass
        
    except PolygonAPIError as e:
        error = get_error_message(
            ErrorType.API_ERROR,
            "rate_limit" if "rate limit" in str(e).lower() else "connection",
            details=str(e)
        )
        errors_encountered.append(error)
        if error_cb:
            error_cb(error)
            
    except RedisConnectionError as e:
        error = get_error_message(
            ErrorType.CACHE_ERROR,
            "connection",
            details=str(e)
        )
        errors_encountered.append(error)
        if error_cb:
            error_cb(error)
        # Continue without cache
        
    except Exception as e:
        error = ScanError(
            error_type=ErrorType.SYSTEM_ERROR,
            message="Unexpected error during scan",
            details=str(e),
            suggestion="Please try again or contact support"
        )
        errors_encountered.append(error)
        if error_cb:
            error_cb(error)
    
    # Add errors to performance metrics
    performance_metrics["errors"] = [e.to_dict() for e in errors_encountered]
    
    return results_df, status_message, performance_metrics
```

### Step 3: Add Progress Time Estimation

**File**: `technic_v4/progress.py` (new file)

```python
"""
Progress tracking utilities with time estimation
"""
import time
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ProgressTracker:
    """Track progress with time estimation"""
    total: int
    start_time: float = field(default_factory=time.time)
    current: int = 0
    
    def update(self, current: int) -> dict:
        """
        Update progress and calculate estimates
        
        Returns:
            Progress information with time estimates
        """
        self.current = current
        elapsed = time.time() - self.start_time
        
        if current == 0:
            return {
                "current": 0,
                "total": self.total,
                "progress_pct": 0.0,
                "elapsed_time": 0.0,
                "estimated_remaining": None,
                "estimated_total": None,
                "throughput": 0.0
            }
        
        # Calculate metrics
        progress_pct = (current / self.total * 100) if self.total > 0 else 0
        throughput = current / elapsed if elapsed > 0 else 0
        
        # Estimate remaining time
        if throughput > 0:
            remaining_items = self.total - current
            estimated_remaining = remaining_items / throughput
            estimated_total = elapsed + estimated_remaining
        else:
            estimated_remaining = None
            estimated_total = None
        
        return {
            "current": current,
            "total": self.total,
            "progress_pct": round(progress_pct, 1),
            "elapsed_time": round(elapsed, 1),
            "estimated_remaining": round(estimated_remaining, 1) if estimated_remaining else None,
            "estimated_total": round(estimated_total, 1) if estimated_total else None,
            "throughput": round(throughput, 2)
        }
```

## Testing Plan

### Test 1: Error Message Validation
```python
# test_error_messages.py
def test_api_rate_limit_error():
    error = get_error_message(ErrorType.API_ERROR, "rate_limit")
    assert error.error_type == ErrorType.API_ERROR
    assert error.recoverable == True
    assert error.retry_after == 60
    assert "wait 60 seconds" in error.suggestion.lower()

def test_cache_connection_error():
    error = get_error_message(ErrorType.CACHE_ERROR, "connection")
    assert error.error_type == ErrorType.CACHE_ERROR
    assert "continuing without cache" in error.suggestion.lower()
```

### Test 2: Progress Time Estimation
```python
# test_progress_tracking.py
def test_progress_estimation():
    tracker = ProgressTracker(total=100)
    
    # Simulate progress
    time.sleep(1)
    progress = tracker.update(25)
    
    assert progress["current"] == 25
    assert progress["progress_pct"] == 25.0
    assert progress["throughput"] > 0
    assert progress["estimated_remaining"] is not None
```

### Test 3: Error Recovery
```python
# test_error_recovery.py
def test_scan_continues_after_cache_error():
    # Simulate cache failure
    with mock.patch('redis.Redis.ping', side_effect=ConnectionError):
        df, msg, metrics = run_scan(config)
        
        # Should complete despite cache error
        assert len(df) > 0
        assert "errors" in metrics
        assert metrics["errors"][0]["error_type"] == "cache_error"
```

## Success Criteria

- ✅ All error types have user-friendly messages
- ✅ Progress includes time estimates
- ✅ Errors include actionable suggestions
- ✅ Scanner continues after recoverable errors
- ✅ Partial results returned on failure
- ✅ All tests passing

## Timeline

- **Day 1**: Error classification system + enhanced messages
- **Day 2**: Progress time estimation + integration
- **Day 3**: Error recovery strategies + testing

## Next Steps After Completion

1. Frontend integration (display errors and progress)
2. Add retry UI controls
3. Implement error analytics
4. Create error documentation for users
