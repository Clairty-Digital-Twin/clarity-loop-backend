"""Cross-cutting concern decorators following GoF Decorator pattern.

Implements decorator pattern for orthogonal concerns like logging, timing,
error handling, and monitoring following Gang of Four design patterns.
"""

import asyncio
import functools
import logging
import time
from datetime import datetime
from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def log_execution(
    level: int = logging.INFO,
    include_args: bool = False,
    include_result: bool = False,
) -> Callable[[F], F]:
    """Decorator to log function execution.
    
    Args:
        level: Logging level (default: INFO)
        include_args: Whether to log function arguments
        include_result: Whether to log function result
        
    Returns:
        Decorated function with logging capabilities
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            # Log function entry
            log_msg = f"Executing {func_name}"
            if include_args:
                log_msg += f" with args={args}, kwargs={kwargs}"
            logger.log(level, log_msg)
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful completion
                completion_msg = f"Completed {func_name}"
                if include_result:
                    completion_msg += f" -> {result}"
                logger.log(level, completion_msg)
                
                return result
            except Exception as e:
                logger.error(f"Error in {func_name}: {e}", exc_info=True)
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            # Log function entry
            log_msg = f"Executing {func_name}"
            if include_args:
                log_msg += f" with args={args}, kwargs={kwargs}"
            logger.log(level, log_msg)
            
            try:
                result = await func(*args, **kwargs)
                
                # Log successful completion
                completion_msg = f"Completed {func_name}"
                if include_result:
                    completion_msg += f" -> {result}"
                logger.log(level, completion_msg)
                
                return result
            except Exception as e:
                logger.error(f"Error in {func_name}: {e}", exc_info=True)
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)
    
    return decorator


def measure_execution_time(
    log_level: int = logging.INFO,
    threshold_ms: float | None = None,
) -> Callable[[F], F]:
    """Decorator to measure and log execution time.
    
    Args:
        log_level: Logging level for timing information
        threshold_ms: Only log if execution time exceeds threshold (milliseconds)
        
    Returns:
        Decorated function with timing capabilities
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = f"{func.__module__}.{func.__qualname__}"
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                
                if threshold_ms is None or execution_time_ms > threshold_ms:
                    logger.log(
                        log_level,
                        f"{func_name} executed in {execution_time_ms:.2f}ms"
                    )
                
                return result
            except Exception:
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                logger.log(
                    log_level,
                    f"{func_name} failed after {execution_time_ms:.2f}ms"
                )
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = f"{func.__module__}.{func.__qualname__}"
            start_time = time.perf_counter()
            
            try:
                result = await func(*args, **kwargs)
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                
                if threshold_ms is None or execution_time_ms > threshold_ms:
                    logger.log(
                        log_level,
                        f"{func_name} executed in {execution_time_ms:.2f}ms"
                    )
                
                return result
            except Exception:
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                logger.log(
                    log_level,
                    f"{func_name} failed after {execution_time_ms:.2f}ms"
                )
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)
    
    return decorator


def retry_on_failure(
    max_retries: int = 3,
    delay_seconds: float = 1.0,
    exponential_backoff: bool = True,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator to retry function execution on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay_seconds: Initial delay between retries
        exponential_backoff: Whether to use exponential backoff
        exceptions: Tuple of exception types to retry on
        
    Returns:
        Decorated function with retry capabilities
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = f"{func.__module__}.{func.__qualname__}"
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"{func_name} failed after {max_retries} retries: {e}"
                        )
                        raise
                    
                    delay = delay_seconds * (2 ** attempt if exponential_backoff else 1)
                    logger.warning(
                        f"{func_name} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = f"{func.__module__}.{func.__qualname__}"
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"{func_name} failed after {max_retries} retries: {e}"
                        )
                        raise
                    
                    delay = delay_seconds * (2 ** attempt if exponential_backoff else 1)
                    logger.warning(
                        f"{func_name} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)
    
    return decorator


def validate_input(validator: Callable[[Any], bool], error_message: str) -> Callable[[F], F]:
    """Decorator to validate function input.
    
    Args:
        validator: Function to validate input arguments
        error_message: Error message to raise if validation fails
        
    Returns:
        Decorated function with input validation
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not validator((args, kwargs)):
                raise ValueError(error_message)
            return func(*args, **kwargs)
        
        return cast(F, wrapper)
    
    return decorator


def audit_trail(
    operation: str,
    user_id_param: str | None = None,
    resource_id_param: str | None = None,
) -> Callable[[F], F]:
    """Decorator to create audit trail for sensitive operations.
    
    Args:
        operation: Name of the operation being audited
        user_id_param: Parameter name containing user ID
        resource_id_param: Parameter name containing resource ID
        
    Returns:
        Decorated function with audit trail capabilities
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract audit information
            audit_info = {
                "operation": operation,
                "timestamp": datetime.utcnow().isoformat(),
                "function": f"{func.__module__}.{func.__qualname__}",
            }
            
            if user_id_param and user_id_param in kwargs:
                audit_info["user_id"] = kwargs[user_id_param]
            
            if resource_id_param and resource_id_param in kwargs:
                audit_info["resource_id"] = kwargs[resource_id_param]
            
            logger.info(f"AUDIT: {audit_info}")
            
            try:
                result = func(*args, **kwargs)
                audit_info["status"] = "success"
                logger.info(f"AUDIT_COMPLETE: {audit_info}")
                return result
            except Exception as e:
                audit_info["status"] = "failed"
                audit_info["error"] = str(e)
                logger.error(f"AUDIT_FAILED: {audit_info}")
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract audit information
            audit_info = {
                "operation": operation,
                "timestamp": datetime.utcnow().isoformat(),
                "function": f"{func.__module__}.{func.__qualname__}",
            }
            
            if user_id_param and user_id_param in kwargs:
                audit_info["user_id"] = kwargs[user_id_param]
            
            if resource_id_param and resource_id_param in kwargs:
                audit_info["resource_id"] = kwargs[resource_id_param]
            
            logger.info(f"AUDIT: {audit_info}")
            
            try:
                result = await func(*args, **kwargs)
                audit_info["status"] = "success"
                logger.info(f"AUDIT_COMPLETE: {audit_info}")
                return result
            except Exception as e:
                audit_info["status"] = "failed"
                audit_info["error"] = str(e)
                logger.error(f"AUDIT_FAILED: {audit_info}")
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)
    
    return decorator


# Convenience combinators for common patterns
def service_method(
    log_level: int = logging.INFO,
    timing_threshold_ms: float | None = 100.0,
    max_retries: int = 0,
) -> Callable[[F], F]:
    """Composite decorator for service layer methods.
    
    Combines logging, timing, and optional retry functionality.
    
    Args:
        log_level: Logging level for execution logs
        timing_threshold_ms: Threshold for logging slow operations
        max_retries: Number of retries for transient failures
        
    Returns:
        Decorated function with service layer concerns
    """
    def decorator(func: F) -> F:
        decorated = func
        
        # Apply decorators in reverse order (innermost first)
        if max_retries > 0:
            decorated = retry_on_failure(max_retries=max_retries)(decorated)
        
        decorated = measure_execution_time(
            log_level=log_level,
            threshold_ms=timing_threshold_ms
        )(decorated)
        
        decorated = log_execution(level=log_level)(decorated)
        
        return decorated
    
    return decorator


def repository_method(
    log_level: int = logging.DEBUG,
    timing_threshold_ms: float | None = 50.0,
    max_retries: int = 2,
) -> Callable[[F], F]:
    """Composite decorator for repository layer methods.
    
    Combines logging, timing, and retry functionality optimized for data access.
    
    Args:
        log_level: Logging level for execution logs
        timing_threshold_ms: Threshold for logging slow operations
        max_retries: Number of retries for transient failures
        
    Returns:
        Decorated function with repository layer concerns
    """
    def decorator(func: F) -> F:
        decorated = func
        
        # Apply decorators in reverse order (innermost first)
        decorated = retry_on_failure(
            max_retries=max_retries,
            delay_seconds=0.5,
            exponential_backoff=True
        )(decorated)
        
        decorated = measure_execution_time(
            log_level=log_level,
            threshold_ms=timing_threshold_ms
        )(decorated)
        
        decorated = log_execution(level=log_level)(decorated)
        
        return decorated
    
    return decorator 