"""Cross-cutting concern decorators following GoF Decorator pattern.

Implements decorator pattern for orthogonal concerns like logging, timing,
error handling, and monitoring following Gang of Four design patterns.
"""

# removed - breaks FastAPI

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
import functools
import logging
import time
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def log_execution(
    level: int = logging.INFO,
    *,
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
            log_msg = "Executing %s"
            log_args = [func_name]
            if include_args:
                log_msg += " with args=%s, kwargs=%s"
                log_args.extend([str(args), str(kwargs)])
            logger.log(level, log_msg, *log_args)

            try:
                result = func(*args, **kwargs)
            except Exception:
                logger.exception("Error in %s", func_name)
                raise
            else:
                # Log successful completion
                completion_msg = "Completed %s"
                completion_args = [func_name]
                if include_result:
                    completion_msg += " -> %s"
                    completion_args.append(result)
                logger.log(level, completion_msg, *completion_args)

                return result

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = f"{func.__module__}.{func.__qualname__}"

            # Log function entry
            log_msg = "Executing %s"
            log_args = [func_name]
            if include_args:
                log_msg += " with args=%s, kwargs=%s"
                log_args.extend([str(args), str(kwargs)])
            logger.log(level, log_msg, *log_args)

            try:
                result = await func(*args, **kwargs)
            except Exception:
                logger.exception("Error in %s", func_name)
                raise
            else:
                # Log successful completion
                completion_msg = "Completed %s"
                completion_args = [func_name]
                if include_result:
                    completion_msg += " -> %s"
                    completion_args.append(result)
                logger.log(level, completion_msg, *completion_args)

                return result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return cast("F", async_wrapper)
        return cast("F", sync_wrapper)

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
            except Exception:
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                logger.log(
                    log_level, "%s failed after %.2fms", func_name, execution_time_ms
                )
                raise
            else:
                execution_time_ms = (time.perf_counter() - start_time) * 1000

                if threshold_ms is None or execution_time_ms > threshold_ms:
                    logger.log(
                        log_level, "%s executed in %.2fms", func_name, execution_time_ms
                    )

                return result

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = f"{func.__module__}.{func.__qualname__}"
            start_time = time.perf_counter()

            try:
                result = await func(*args, **kwargs)
            except Exception:
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                logger.log(
                    log_level, "%s failed after %.2fms", func_name, execution_time_ms
                )
                raise
            else:
                execution_time_ms = (time.perf_counter() - start_time) * 1000

                if threshold_ms is None or execution_time_ms > threshold_ms:
                    logger.log(
                        log_level, "%s executed in %.2fms", func_name, execution_time_ms
                    )

                return result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return cast("F", async_wrapper)
        return cast("F", sync_wrapper)

    return decorator


def retry_on_failure(
    max_retries: int = 3,
    delay_seconds: float = 1.0,
    *,
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
                        logger.exception(
                            "Function %s failed after %d attempts",
                            func_name,
                            max_retries + 1,
                        )
                        raise

                    # Calculate delay for next attempt
                    current_delay = delay_seconds
                    if exponential_backoff:
                        current_delay *= 2**attempt

                    logger.warning(
                        "Function %s failed (attempt %d/%d), retrying in %.2fs",
                        func_name,
                        attempt + 1,
                        max_retries + 1,
                        current_delay,
                    )
                    time.sleep(current_delay)

            # This should never be reached, but satisfy type checker
            if last_exception:
                raise last_exception
            return None

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
                        logger.exception(
                            "Function %s failed after %d attempts",
                            func_name,
                            max_retries + 1,
                        )
                        raise

                    # Calculate delay for next attempt
                    current_delay = delay_seconds
                    if exponential_backoff:
                        current_delay *= 2**attempt

                    logger.warning(
                        "Function %s failed (attempt %d/%d), retrying in %.2fs",
                        func_name,
                        attempt + 1,
                        max_retries + 1,
                        current_delay,
                    )
                    await asyncio.sleep(current_delay)

            # This should never be reached, but satisfy type checker
            if last_exception:
                raise last_exception
            return None

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return cast("F", async_wrapper)
        return cast("F", sync_wrapper)

    return decorator


def validate_input(
    validator: Callable[[Any], bool], error_message: str
) -> Callable[[F], F]:
    """Decorator to validate function input.

    Args:
        validator: Function to validate input (args, kwargs)
        error_message: Error message to raise on validation failure

    Returns:
        Decorated function with input validation
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not validator((args, kwargs)):
                raise ValueError(error_message)
            return func(*args, **kwargs)

        return cast("F", wrapper)

    return decorator


def audit_trail(
    operation: str,
    user_id_param: str | None = None,
    resource_id_param: str | None = None,
) -> Callable[[F], F]:
    """Decorator to create audit trail for function execution.

    Args:
        operation: Description of the operation being performed
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
                "function": f"{func.__module__}.{func.__qualname__}",
                "timestamp": datetime.now(UTC).isoformat(),
                "user_id": kwargs.get(user_id_param) if user_id_param else None,
                "resource_id": (
                    kwargs.get(resource_id_param) if resource_id_param else None
                ),
            }

            logger.info("Audit: %s", audit_info)

            try:
                result = func(*args, **kwargs)
            except Exception as e:
                audit_info["status"] = "failed"
                audit_info["error"] = str(e)
                logger.exception("Audit failed: %s", audit_info)
                raise
            else:
                audit_info["status"] = "success"
                logger.info("Audit completed: %s", audit_info)
                return result

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract audit information
            audit_info = {
                "operation": operation,
                "function": f"{func.__module__}.{func.__qualname__}",
                "timestamp": datetime.now(UTC).isoformat(),
                "user_id": kwargs.get(user_id_param) if user_id_param else None,
                "resource_id": (
                    kwargs.get(resource_id_param) if resource_id_param else None
                ),
            }

            logger.info("Audit: %s", audit_info)

            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                audit_info["status"] = "failed"
                audit_info["error"] = str(e)
                logger.exception("Audit failed: %s", audit_info)
                raise
            else:
                audit_info["status"] = "success"
                logger.info("Audit completed: %s", audit_info)
                return result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return cast("F", async_wrapper)
        return cast("F", sync_wrapper)

    return decorator


def service_method(
    log_level: int = logging.INFO,
    timing_threshold_ms: float | None = 100.0,
    max_retries: int = 0,
) -> Callable[[F], F]:
    """Composite decorator for service layer methods.

    Combines logging, timing, and optional retry functionality.

    Args:
        log_level: Logging level for execution logs
        timing_threshold_ms: Threshold for logging execution time
        max_retries: Number of retry attempts (0 = no retries)

    Returns:
        Decorated function with service method capabilities
    """

    def decorator(func: F) -> F:
        decorated = func
        decorated = log_execution(level=log_level)(decorated)
        decorated = measure_execution_time(
            log_level=log_level, threshold_ms=timing_threshold_ms
        )(decorated)

        if max_retries > 0:
            decorated = retry_on_failure(max_retries=max_retries)(decorated)

        return log_execution(level=log_level)(decorated)

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
        timing_threshold_ms: Threshold for logging execution time
        max_retries: Number of retry attempts

    Returns:
        Decorated function with repository method capabilities
    """

    def decorator(func: F) -> F:
        decorated = func
        decorated = log_execution(level=log_level)(decorated)
        decorated = measure_execution_time(
            log_level=log_level, threshold_ms=timing_threshold_ms
        )(decorated)
        decorated = retry_on_failure(max_retries=max_retries)(decorated)
        return log_execution(level=log_level)(decorated)

    return decorator
