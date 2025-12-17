"""
Enhanced Error Handler Component
User-friendly error display with retry mechanisms and recovery options
Path 1 Task 3: Error Handling
"""

import streamlit as st
import time
from typing import Callable, Optional, Any, Dict
from functools import wraps
import traceback
import sys
from pathlib import Path

# Add technic_v4 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from technic_v4.errors import ScanError, ErrorType, get_error_message, create_custom_error


class ErrorHandler:
    """
    Enhanced error handler with retry mechanisms and user-friendly display
    
    Features:
    - User-friendly error messages
    - Automatic retry with exponential backoff
    - Manual retry buttons
    - Error logging
    - Recovery suggestions
    - Fallback strategies
    """
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize error handler
        
        Args:
            max_retries: Maximum number of automatic retries
            base_delay: Base delay in seconds for exponential backoff
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.error_history = []
    
    def retry_with_backoff(
        self,
        func: Callable,
        *args,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with exponential backoff retry
        
        Args:
            func: Function to execute
            *args: Function arguments
            max_retries: Override default max retries
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Last exception if all retries fail
        """
        max_attempts = max_retries or self.max_retries
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < max_attempts - 1:
                    # Calculate delay with exponential backoff
                    delay = self.base_delay * (2 ** attempt)
                    
                    # Log retry attempt
                    self.error_history.append({
                        'attempt': attempt + 1,
                        'error': str(e),
                        'retry_delay': delay
                    })
                    
                    time.sleep(delay)
                else:
                    # Final attempt failed
                    raise last_exception
        
        raise last_exception
    
    def display_error(
        self,
        error: ScanError,
        show_details: bool = True,
        show_retry: bool = True,
        retry_callback: Optional[Callable] = None
    ):
        """
        Display user-friendly error message in Streamlit
        
        Args:
            error: ScanError instance
            show_details: Whether to show technical details
            show_retry: Whether to show retry button
            retry_callback: Function to call on retry
        """
        # Determine error severity
        if error.error_type in [ErrorType.CONFIG_ERROR, ErrorType.SYSTEM_ERROR]:
            severity = "error"
        elif error.error_type in [ErrorType.TIMEOUT_ERROR, ErrorType.API_ERROR]:
            severity = "warning"
        else:
            severity = "info"
        
        # Display main error message
        if severity == "error":
            st.error(f"❌ **{error.message}**")
        elif severity == "warning":
            st.warning(f"⚠️ **{error.message}**")
        else:
            st.info(f"ℹ️ **{error.message}**")
        
        # Display suggestion
        st.markdown(f"**💡 Suggestion:** {error.suggestion}")
        
        # Show affected symbols if any
        if error.affected_symbols:
            with st.expander("📋 Affected Symbols"):
                st.write(", ".join(error.affected_symbols))
        
        # Show technical details
        if show_details:
            with st.expander("🔍 Technical Details"):
                st.code(error.details)
                
                if error.metadata:
                    st.json(error.metadata)
        
        # Show retry options
        if show_retry and error.recoverable:
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("🔄 Retry", key=f"retry_{id(error)}"):
                    if retry_callback:
                        retry_callback()
                    else:
                        st.rerun()
            
            with col2:
                if error.retry_after:
                    st.caption(f"Retry after: {error.retry_after}s")
            
            with col3:
                if st.button("❌ Cancel", key=f"cancel_{id(error)}"):
                    st.stop()
    
    def display_error_from_exception(
        self,
        exception: Exception,
        context: str = "",
        show_retry: bool = True,
        retry_callback: Optional[Callable] = None
    ):
        """
        Convert exception to ScanError and display
        
        Args:
            exception: Python exception
            context: Context where error occurred
            show_retry: Whether to show retry button
            retry_callback: Function to call on retry
        """
        # Map common exceptions to error types
        error_mapping = {
            ConnectionError: (ErrorType.API_ERROR, "connection"),
            TimeoutError: (ErrorType.TIMEOUT_ERROR, "scan"),
            ValueError: (ErrorType.DATA_ERROR, "invalid"),
            KeyError: (ErrorType.CONFIG_ERROR, "missing"),
        }
        
        # Get error type and key
        error_type, error_key = error_mapping.get(
            type(exception),
            (ErrorType.SYSTEM_ERROR, "unknown")
        )
        
        # Create error message
        error = get_error_message(
            error_type,
            error_key,
            details=f"{context}: {str(exception)}\n\n{traceback.format_exc()}",
            metadata={'exception_type': type(exception).__name__}
        )
        
        # Display error
        self.display_error(error, show_retry=show_retry, retry_callback=retry_callback)
    
    def safe_execute(
        self,
        func: Callable,
        *args,
        error_message: str = "Operation failed",
        show_retry: bool = True,
        **kwargs
    ) -> Optional[Any]:
        """
        Safely execute function with error handling
        
        Args:
            func: Function to execute
            *args: Function arguments
            error_message: Custom error message
            show_retry: Whether to show retry button
            **kwargs: Function keyword arguments
        
        Returns:
            Function result or None if error
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.display_error_from_exception(
                e,
                context=error_message,
                show_retry=show_retry,
                retry_callback=lambda: func(*args, **kwargs)
            )
            return None
    
    def with_fallback(
        self,
        primary_func: Callable,
        fallback_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with fallback on error
        
        Args:
            primary_func: Primary function to try
            fallback_func: Fallback function if primary fails
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Result from primary or fallback function
        """
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            st.warning(f"⚠️ Primary method failed, using fallback: {str(e)}")
            return fallback_func(*args, **kwargs)


def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator for automatic retry with exponential backoff
    
    Args:
        max_retries: Maximum number of retries
        delay: Base delay in seconds
    
    Example:
        @retry_on_error(max_retries=3, delay=2.0)
        def fetch_data(symbol):
            return api.get_data(symbol)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = ErrorHandler(max_retries=max_retries, base_delay=delay)
            return handler.retry_with_backoff(func, *args, **kwargs)
        return wrapper
    return decorator


def safe_api_call(func):
    """
    Decorator for safe API calls with user-friendly error handling
    
    Example:
        @safe_api_call
        def get_stock_data(symbol):
            return api.fetch(symbol)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConnectionError as e:
            error = get_error_message(
                ErrorType.API_ERROR,
                "connection",
                details=str(e)
            )
            st.error(f"❌ {error.message}")
            st.info(f"💡 {error.suggestion}")
            return None
        except TimeoutError as e:
            error = get_error_message(
                ErrorType.TIMEOUT_ERROR,
                "scan",
                details=str(e)
            )
            st.warning(f"⚠️ {error.message}")
            st.info(f"💡 {error.suggestion}")
            return None
        except Exception as e:
            error = create_custom_error(
                ErrorType.SYSTEM_ERROR,
                "Unexpected error",
                str(e),
                "Please try again or contact support"
            )
            st.error(f"❌ {error.message}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            return None
    return wrapper


# Example usage functions
def example_basic_error():
    """Example: Basic error display"""
    st.header("Example: Basic Error Display")
    
    if st.button("Trigger API Error"):
        error = get_error_message(
            ErrorType.API_ERROR,
            "rate_limit",
            affected_symbols=["AAPL", "MSFT", "GOOGL"]
        )
        
        handler = ErrorHandler()
        handler.display_error(error)


def example_retry_mechanism():
    """Example: Retry with backoff"""
    st.header("Example: Retry Mechanism")
    
    if st.button("Test Retry Logic"):
        attempt_count = [0]
        
        def failing_function():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise ConnectionError(f"Attempt {attempt_count[0]} failed")
            return "Success!"
        
        handler = ErrorHandler(max_retries=3, base_delay=0.5)
        
        try:
            with st.spinner("Attempting operation with retry..."):
                result = handler.retry_with_backoff(failing_function)
            st.success(f"✅ {result} (after {attempt_count[0]} attempts)")
        except Exception as e:
            st.error(f"❌ All retries failed: {str(e)}")


def example_safe_execution():
    """Example: Safe execution with error handling"""
    st.header("Example: Safe Execution")
    
    if st.button("Execute with Error Handling"):
        handler = ErrorHandler()
        
        def risky_operation():
            import random
            if random.random() < 0.5:
                raise ValueError("Random error occurred!")
            return "Operation successful!"
        
        result = handler.safe_execute(
            risky_operation,
            error_message="Risky operation",
            show_retry=True
        )
        
        if result:
            st.success(f"✅ {result}")


def example_fallback():
    """Example: Fallback strategy"""
    st.header("Example: Fallback Strategy")
    
    if st.button("Test Fallback"):
        handler = ErrorHandler()
        
        def primary_method():
            raise ConnectionError("Primary API unavailable")
        
        def fallback_method():
            return "Using cached data instead"
        
        result = handler.with_fallback(
            primary_method,
            fallback_method
        )
        
        st.info(f"ℹ️ Result: {result}")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Error Handler Examples",
        page_icon="❌",
        layout="wide"
    )
    
    st.title("❌ Error Handler Component")
    st.markdown("Examples of enhanced error handling with retry mechanisms")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Basic Errors",
        "Retry Mechanism",
        "Safe Execution",
        "Fallback Strategy"
    ])
    
    with tab1:
        example_basic_error()
    
    with tab2:
        example_retry_mechanism()
    
    with tab3:
        example_safe_execution()
    
    with tab4:
        example_fallback()
    
    # Documentation
    st.markdown("---")
    st.header("📚 Usage Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Basic Error Display:**
        ```python
        from components.ErrorHandler import ErrorHandler
        from technic_v4.errors import get_error_message, ErrorType
        
        error = get_error_message(
            ErrorType.API_ERROR,
            "rate_limit"
        )
        
        handler = ErrorHandler()
        handler.display_error(error)
        ```
        """)
    
    with col2:
        st.markdown("""
        **Retry Decorator:**
        ```python
        from components.ErrorHandler import retry_on_error
        
        @retry_on_error(max_retries=3, delay=2.0)
        def fetch_data(symbol):
            return api.get_data(symbol)
        
        result = fetch_data("AAPL")
        ```
        """)
