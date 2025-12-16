"""
Test error message system for Phase 3D-B
"""
import pytest
from technic_v4.errors import (
    ErrorType,
    ScanError,
    get_error_message,
    create_custom_error
)


def test_api_rate_limit_error():
    """Test API rate limit error message"""
    error = get_error_message(ErrorType.API_ERROR, "rate_limit")
    
    assert error.error_type == ErrorType.API_ERROR
    assert error.recoverable == True
    assert error.retry_after == 60
    assert "rate limit" in error.message.lower()
    assert "wait 60 seconds" in error.suggestion.lower()
    
    # Test to_dict conversion
    error_dict = error.to_dict()
    assert error_dict["error_type"] == "api_error"
    assert error_dict["retry_after"] == 60
    print("✅ API rate limit error test passed")


def test_api_connection_error():
    """Test API connection error message"""
    error = get_error_message(ErrorType.API_ERROR, "connection")
    
    assert error.error_type == ErrorType.API_ERROR
    assert "connect" in error.message.lower()
    assert "internet connection" in error.suggestion.lower()
    assert error.recoverable == True
    print("✅ API connection error test passed")


def test_api_invalid_key_error():
    """Test API invalid key error (non-recoverable)"""
    error = get_error_message(ErrorType.API_ERROR, "invalid_key")
    
    assert error.error_type == ErrorType.API_ERROR
    assert error.recoverable == False
    assert "api key" in error.message.lower()
    assert "POLYGON_API_KEY" in error.suggestion
    print("✅ API invalid key error test passed")


def test_cache_connection_error():
    """Test cache connection error message"""
    error = get_error_message(ErrorType.CACHE_ERROR, "connection")
    
    assert error.error_type == ErrorType.CACHE_ERROR
    assert "cache" in error.message.lower()
    assert "continuing without cache" in error.suggestion.lower()
    assert error.recoverable == True
    print("✅ Cache connection error test passed")


def test_cache_timeout_error():
    """Test cache timeout error message"""
    error = get_error_message(ErrorType.CACHE_ERROR, "timeout")
    
    assert error.error_type == ErrorType.CACHE_ERROR
    assert "timeout" in error.message.lower()
    assert "bypassed" in error.suggestion.lower()
    print("✅ Cache timeout error test passed")


def test_data_missing_error():
    """Test missing data error message"""
    error = get_error_message(ErrorType.DATA_ERROR, "missing")
    
    assert error.error_type == ErrorType.DATA_ERROR
    assert "insufficient" in error.message.lower() or "missing" in error.message.lower()
    assert "skipped" in error.suggestion.lower()
    print("✅ Data missing error test passed")


def test_data_invalid_error():
    """Test invalid data error message"""
    error = get_error_message(ErrorType.DATA_ERROR, "invalid")
    
    assert error.error_type == ErrorType.DATA_ERROR
    assert "invalid" in error.message.lower()
    assert "skipped" in error.suggestion.lower()
    print("✅ Data invalid error test passed")


def test_timeout_scan_error():
    """Test scan timeout error message"""
    error = get_error_message(ErrorType.TIMEOUT_ERROR, "scan")
    
    assert error.error_type == ErrorType.TIMEOUT_ERROR
    assert "timeout" in error.message.lower()
    assert "reducing" in error.suggestion.lower() or "symbols" in error.suggestion.lower()
    print("✅ Scan timeout error test passed")


def test_config_invalid_error():
    """Test invalid config error (non-recoverable)"""
    error = get_error_message(ErrorType.CONFIG_ERROR, "invalid")
    
    assert error.error_type == ErrorType.CONFIG_ERROR
    assert error.recoverable == False
    assert "configuration" in error.message.lower()
    assert "settings" in error.suggestion.lower()
    print("✅ Config invalid error test passed")


def test_error_with_affected_symbols():
    """Test error with affected symbols list"""
    error = get_error_message(
        ErrorType.API_ERROR,
        "rate_limit",
        affected_symbols=["AAPL", "MSFT", "GOOGL"]
    )
    
    assert error.affected_symbols == ["AAPL", "MSFT", "GOOGL"]
    
    error_dict = error.to_dict()
    assert "affected_symbols" in error_dict
    assert len(error_dict["affected_symbols"]) == 3
    print("✅ Error with affected symbols test passed")


def test_error_with_metadata():
    """Test error with custom metadata"""
    metadata = {
        "api_endpoint": "/v2/aggs/ticker",
        "status_code": 429,
        "timestamp": "2025-12-16T12:00:00Z"
    }
    
    error = get_error_message(
        ErrorType.API_ERROR,
        "rate_limit",
        metadata=metadata
    )
    
    assert error.metadata == metadata
    
    error_dict = error.to_dict()
    assert "metadata" in error_dict
    assert error_dict["metadata"]["status_code"] == 429
    print("✅ Error with metadata test passed")


def test_error_with_custom_details():
    """Test overriding error details"""
    custom_details = "Specific error: Connection refused on port 443"
    
    error = get_error_message(
        ErrorType.API_ERROR,
        "connection",
        details=custom_details
    )
    
    assert error.details == custom_details
    print("✅ Error with custom details test passed")


def test_create_custom_error():
    """Test creating a fully custom error"""
    error = create_custom_error(
        error_type=ErrorType.DATA_ERROR,
        message="Symbol not found",
        details="INVALID_SYMBOL does not exist in the universe",
        suggestion="Check the symbol ticker and try again",
        affected_symbols=["INVALID_SYMBOL"],
        recoverable=True
    )
    
    assert error.error_type == ErrorType.DATA_ERROR
    assert error.message == "Symbol not found"
    assert "INVALID_SYMBOL" in error.details
    assert error.recoverable == True
    assert error.affected_symbols == ["INVALID_SYMBOL"]
    print("✅ Create custom error test passed")


def test_unknown_error_fallback():
    """Test fallback for unknown error types"""
    error = get_error_message(
        ErrorType.SYSTEM_ERROR,
        "nonexistent_error_key",
        details="This is a test error"
    )
    
    assert error.error_type == ErrorType.SYSTEM_ERROR
    assert "This is a test error" in error.details
    print("✅ Unknown error fallback test passed")


def test_error_to_dict_complete():
    """Test complete error serialization"""
    error = create_custom_error(
        error_type=ErrorType.API_ERROR,
        message="Test error",
        details="Test details",
        suggestion="Test suggestion",
        recoverable=True,
        retry_after=30,
        affected_symbols=["TEST"],
        metadata={"test_key": "test_value"}
    )
    
    error_dict = error.to_dict()
    
    assert error_dict["error_type"] == "api_error"
    assert error_dict["message"] == "Test error"
    assert error_dict["details"] == "Test details"
    assert error_dict["suggestion"] == "Test suggestion"
    assert error_dict["recoverable"] == True
    assert error_dict["retry_after"] == 30
    assert error_dict["affected_symbols"] == ["TEST"]
    assert error_dict["metadata"]["test_key"] == "test_value"
    print("✅ Error to_dict complete test passed")


def test_all_error_types_have_messages():
    """Test that all error types have at least one predefined message"""
    from technic_v4.errors import ERROR_MESSAGES
    
    for error_type in ErrorType:
        assert error_type in ERROR_MESSAGES, f"Missing messages for {error_type}"
        assert len(ERROR_MESSAGES[error_type]) > 0, f"No messages defined for {error_type}"
    
    print("✅ All error types have messages test passed")


if __name__ == "__main__":
    print("="*80)
    print("TESTING ERROR MESSAGE SYSTEM")
    print("="*80)
    print()
    
    # Run all tests
    test_api_rate_limit_error()
    test_api_connection_error()
    test_api_invalid_key_error()
    test_cache_connection_error()
    test_cache_timeout_error()
    test_data_missing_error()
    test_data_invalid_error()
    test_timeout_scan_error()
    test_config_invalid_error()
    test_error_with_affected_symbols()
    test_error_with_metadata()
    test_error_with_custom_details()
    test_create_custom_error()
    test_unknown_error_fallback()
    test_error_to_dict_complete()
    test_all_error_types_have_messages()
    
    print()
    print("="*80)
    print("✅ ALL ERROR MESSAGE TESTS PASSED!")
    print("="*80)
