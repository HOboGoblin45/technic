"""
Enhanced error handling for Technic Scanner
Provides user-friendly error messages with actionable suggestions
"""
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
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
    """Structured error information with user-friendly messages"""
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
        result = {
            "error_type": self.error_type.value if isinstance(self.error_type, ErrorType) else self.error_type,
            "message": self.message,
            "details": self.details,
            "suggestion": self.suggestion,
            "recoverable": self.recoverable,
        }
        
        if self.retry_after is not None:
            result["retry_after"] = self.retry_after
        if self.affected_symbols:
            result["affected_symbols"] = self.affected_symbols
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result


# Predefined error messages for common scenarios
ERROR_MESSAGES = {
    ErrorType.API_ERROR: {
        "rate_limit": ScanError(
            error_type=ErrorType.API_ERROR,
            message="API rate limit exceeded",
            details="Too many requests to Polygon API",
            suggestion="Please wait 60 seconds and try again, or upgrade your API plan for higher limits",
            retry_after=60
        ),
        "connection": ScanError(
            error_type=ErrorType.API_ERROR,
            message="Unable to connect to market data provider",
            details="Network connection to Polygon API failed",
            suggestion="Check your internet connection and try again in a few moments"
        ),
        "invalid_key": ScanError(
            error_type=ErrorType.API_ERROR,
            message="Invalid API key",
            details="Polygon API key is invalid or expired",
            suggestion="Check your POLYGON_API_KEY environment variable and ensure it's valid",
            recoverable=False
        ),
        "timeout": ScanError(
            error_type=ErrorType.API_ERROR,
            message="API request timeout",
            details="Market data request took too long to complete",
            suggestion="The API may be experiencing high load. Try again in a few moments."
        )
    },
    ErrorType.CACHE_ERROR: {
        "connection": ScanError(
            error_type=ErrorType.CACHE_ERROR,
            message="Cache unavailable",
            details="Unable to connect to Redis cache",
            suggestion="Continuing without cache (slower performance). Check Redis connection if this persists."
        ),
        "timeout": ScanError(
            error_type=ErrorType.CACHE_ERROR,
            message="Cache operation timeout",
            details="Redis operation took too long to complete",
            suggestion="Cache will be bypassed for this scan. Consider checking Redis performance."
        ),
        "write_error": ScanError(
            error_type=ErrorType.CACHE_ERROR,
            message="Unable to write to cache",
            details="Failed to store data in Redis cache",
            suggestion="Scan will continue but results won't be cached. Check Redis storage capacity."
        )
    },
    ErrorType.DATA_ERROR: {
        "missing": ScanError(
            error_type=ErrorType.DATA_ERROR,
            message="Insufficient data",
            details="Not enough historical data available for analysis",
            suggestion="Symbol will be skipped. Try increasing the lookback period or check if the symbol is newly listed."
        ),
        "invalid": ScanError(
            error_type=ErrorType.DATA_ERROR,
            message="Invalid data format",
            details="Market data format is unexpected or corrupted",
            suggestion="Symbol will be skipped. This may indicate a data provider issue."
        ),
        "empty": ScanError(
            error_type=ErrorType.DATA_ERROR,
            message="No data available",
            details="No market data returned for the requested period",
            suggestion="Symbol may be delisted or data unavailable. Symbol will be skipped."
        )
    },
    ErrorType.TIMEOUT_ERROR: {
        "scan": ScanError(
            error_type=ErrorType.TIMEOUT_ERROR,
            message="Scan timeout",
            details="Scan operation exceeded the maximum allowed time",
            suggestion="Try reducing the number of symbols, decreasing lookback period, or increasing timeout limit"
        ),
        "symbol": ScanError(
            error_type=ErrorType.TIMEOUT_ERROR,
            message="Symbol analysis timeout",
            details="Individual symbol analysis took too long",
            suggestion="Symbol will be skipped. This may indicate complex calculations or slow data retrieval."
        )
    },
    ErrorType.CONFIG_ERROR: {
        "invalid": ScanError(
            error_type=ErrorType.CONFIG_ERROR,
            message="Invalid configuration",
            details="Scan configuration contains invalid parameters",
            suggestion="Check your scan settings (lookback days, max symbols, etc.) and try again",
            recoverable=False
        ),
        "missing": ScanError(
            error_type=ErrorType.CONFIG_ERROR,
            message="Missing required configuration",
            details="Required configuration parameters are not set",
            suggestion="Ensure all required environment variables and settings are configured",
            recoverable=False
        )
    },
    ErrorType.SYSTEM_ERROR: {
        "unknown": ScanError(
            error_type=ErrorType.SYSTEM_ERROR,
            message="An unexpected error occurred",
            details="Unknown system error",
            suggestion="Please try again. If the problem persists, contact support with error details."
        ),
        "memory": ScanError(
            error_type=ErrorType.SYSTEM_ERROR,
            message="Insufficient memory",
            details="System ran out of available memory",
            suggestion="Try reducing the number of symbols or restart the application"
        )
    }
}


def get_error_message(
    error_type: ErrorType,
    error_key: str,
    **kwargs
) -> ScanError:
    """
    Get a predefined error message with optional customization
    
    Args:
        error_type: Type of error (API_ERROR, CACHE_ERROR, etc.)
        error_key: Specific error within the type (rate_limit, connection, etc.)
        **kwargs: Additional fields to override (details, affected_symbols, etc.)
    
    Returns:
        ScanError instance with user-friendly message
    
    Example:
        >>> error = get_error_message(
        ...     ErrorType.API_ERROR,
        ...     "rate_limit",
        ...     affected_symbols=["AAPL", "MSFT"]
        ... )
        >>> print(error.message)
        API rate limit exceeded
    """
    # Get predefined error or create generic one
    base_error = ERROR_MESSAGES.get(error_type, {}).get(error_key)
    
    if not base_error:
        # Fallback to generic error
        base_error = ScanError(
            error_type=error_type,
            message=f"Error: {error_key}",
            details=str(kwargs.get("details", "Unknown error")),
            suggestion="Please try again or contact support if the problem persists"
        )
    
    # If no overrides, return the base error
    if not kwargs:
        return base_error
    
    # Create a new error with overrides
    error_data = {
        "error_type": kwargs.get("error_type", base_error.error_type),
        "message": kwargs.get("message", base_error.message),
        "details": kwargs.get("details", base_error.details),
        "suggestion": kwargs.get("suggestion", base_error.suggestion),
        "recoverable": kwargs.get("recoverable", base_error.recoverable),
        "retry_after": kwargs.get("retry_after", base_error.retry_after),
        "affected_symbols": kwargs.get("affected_symbols", base_error.affected_symbols),
        "metadata": kwargs.get("metadata", base_error.metadata)
    }
    
    return ScanError(**error_data)


def create_custom_error(
    error_type: ErrorType,
    message: str,
    details: str,
    suggestion: str,
    **kwargs
) -> ScanError:
    """
    Create a custom error message
    
    Args:
        error_type: Type of error
        message: Short error message
        details: Detailed error description
        suggestion: Actionable suggestion for the user
        **kwargs: Additional fields (recoverable, retry_after, etc.)
    
    Returns:
        ScanError instance
    
    Example:
        >>> error = create_custom_error(
        ...     ErrorType.DATA_ERROR,
        ...     "Symbol not found",
        ...     "INVALID_SYMBOL does not exist",
        ...     "Check the symbol ticker and try again",
        ...     affected_symbols=["INVALID_SYMBOL"]
        ... )
    """
    return ScanError(
        error_type=error_type,
        message=message,
        details=details,
        suggestion=suggestion,
        **kwargs
    )
