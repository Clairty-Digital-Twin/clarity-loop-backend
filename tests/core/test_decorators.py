"""Tests for core decorators functionality.

Comprehensive tests for production decorators including:
- Log execution decorator
- Execution time measurement
- Retry patterns with exponential backoff
- Input validation
- Audit trail functionality
- Composite decorators for service and repository layers
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import functools
import logging
import time
from typing import Any, TypeVar

import pytest

from clarity.core.decorators import (
    audit_trail,
    log_execution,
    measure_execution_time,
    repository_method,
    retry_on_failure,
    service_method,
    validate_input,
)

F = TypeVar("F", bound=Callable[..., Any])


class TestLogExecution:
    """Test log_execution decorator."""

    @staticmethod
    def test_log_execution_sync_function_basic(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test basic logging for sync functions."""
        # Set caplog level for the specific logger used by decorators
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        @log_execution()
        def test_function() -> str:
            return "test_result"

        result = test_function()

        assert result == "test_result"
        log_messages = caplog.messages
        assert any(
            "Executing" in msg and "test_function" in msg for msg in log_messages
        )
        assert any(
            "Completed" in msg and "test_function" in msg for msg in log_messages
        )

    @staticmethod
    def test_log_execution_with_args_and_result(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test logging with arguments and result included."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        @log_execution(include_args=True, include_result=True)
        def test_function(x: int, y: str = "default") -> str:
            return f"{x}_{y}"

        result = test_function(42, y="test")

        assert result == "42_test"
        log_messages = " ".join(caplog.messages)
        assert "with args=" in log_messages
        assert "kwargs=" in log_messages
        assert "42_test" in log_messages

    @staticmethod
    async def test_log_execution_async_function(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test logging for async functions."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        @log_execution()
        async def async_test_function() -> str:
            await asyncio.sleep(0.01)
            return "async_result"

        result = await async_test_function()

        assert result == "async_result"
        log_messages = caplog.messages
        assert any(
            "Executing" in msg and "async_test_function" in msg for msg in log_messages
        )
        assert any(
            "Completed" in msg and "async_test_function" in msg for msg in log_messages
        )

    @staticmethod
    def test_log_execution_exception_handling(caplog: pytest.LogCaptureFixture) -> None:
        """Test logging when function raises exception."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        @log_execution()
        def failing_function() -> str:
            msg = "Test error"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

        log_messages = " ".join(caplog.messages)
        assert "Executing" in log_messages
        assert "failing_function" in log_messages
        assert "Error in" in log_messages
        assert "failing_function" in log_messages

    @staticmethod
    async def test_log_execution_async_exception_handling(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test logging when async function raises exception."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        @log_execution()
        async def failing_async_function() -> str:
            await asyncio.sleep(0.01)
            msg = "Async test error"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="Async test error"):
            await failing_async_function()

        log_messages = " ".join(caplog.messages)
        assert "Executing" in log_messages
        assert "failing_async_function" in log_messages
        assert "Error in" in log_messages
        assert "failing_async_function" in log_messages

    @staticmethod
    def test_log_execution_custom_level(caplog: pytest.LogCaptureFixture) -> None:
        """Test logging with custom log level."""
        caplog.set_level(logging.DEBUG, logger="clarity.core.decorators")

        @log_execution(level=logging.DEBUG)
        def debug_function() -> str:
            return "debug_result"

        result = debug_function()

        assert result == "debug_result"
        # Check that debug messages are captured
        debug_messages = [
            record for record in caplog.records if record.levelno == logging.DEBUG
        ]
        assert len(debug_messages) >= 2  # At least entry and completion


class TestMeasureExecutionTime:
    """Test measure_execution_time decorator."""

    @staticmethod
    def test_measure_execution_time_basic(caplog: pytest.LogCaptureFixture) -> None:
        """Test basic execution time measurement."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        @measure_execution_time()
        def timed_function() -> str:
            time.sleep(0.05)  # 50ms delay
            return "timed_result"

        result = timed_function()

        assert result == "timed_result"
        log_messages = " ".join(caplog.messages)
        assert "executed in" in log_messages
        assert "ms" in log_messages

    @staticmethod
    def test_measure_execution_time_with_threshold(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test execution time measurement with threshold."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        @measure_execution_time(threshold_ms=100.0)  # 100ms threshold
        def fast_function() -> str:
            time.sleep(0.01)  # 10ms delay (below threshold)
            return "fast_result"

        @measure_execution_time(threshold_ms=10.0)  # 10ms threshold
        def slow_function() -> str:
            time.sleep(0.05)  # 50ms delay (above threshold)
            return "slow_result"

        fast_result = fast_function()
        slow_result = slow_function()

        assert fast_result == "fast_result"
        assert slow_result == "slow_result"

        log_messages = " ".join(caplog.messages)
        # Fast function should not be logged (below threshold)
        assert "fast_function" not in log_messages
        # Slow function should be logged (above threshold)
        assert "slow_function" in log_messages

    @staticmethod
    async def test_measure_execution_time_async(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test execution time measurement for async functions."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        @measure_execution_time()
        async def async_timed_function() -> str:
            await asyncio.sleep(0.05)  # 50ms delay
            return "async_timed_result"

        result = await async_timed_function()

        assert result == "async_timed_result"
        log_messages = " ".join(caplog.messages)
        assert "async_timed_function executed in" in log_messages

    @staticmethod
    def test_measure_execution_time_exception_handling(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test timing measurement when function raises exception."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        @measure_execution_time()
        def failing_timed_function() -> str:
            time.sleep(0.02)  # Small delay before failing
            msg = "Timing test error"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="Timing test error"):
            failing_timed_function()

        log_messages = " ".join(caplog.messages)
        assert "failed after" in log_messages
        assert "ms" in log_messages


class TestRetryOnFailure:
    """Test retry_on_failure decorator."""

    @staticmethod
    def test_retry_on_failure_success_on_first_try() -> None:
        """Test retry decorator when function succeeds on first try."""
        call_count = 0

        @retry_on_failure(max_retries=3)
        def successful_function() -> str:
            nonlocal call_count
            call_count += 1
            return f"success_{call_count}"

        result = successful_function()
        assert result == "success_1"
        assert call_count == 1

    @staticmethod
    def test_retry_on_failure_success_after_retries(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test retry decorator when function succeeds after retries."""
        call_count = 0

        caplog.set_level(logging.WARNING, logger="clarity.core.decorators")

        with caplog.at_level(logging.WARNING):

            @retry_on_failure(max_retries=3, delay_seconds=0.01)
            def flaky_function() -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    msg = f"Attempt {call_count} failed"
                    raise ValueError(msg)
                return f"success_{call_count}"

            result = flaky_function()

        assert result == "success_3"
        assert call_count == 3
        log_messages = " ".join(caplog.messages)
        assert "retrying in" in log_messages

    @staticmethod
    def test_retry_on_failure_final_failure(caplog: pytest.LogCaptureFixture) -> None:
        """Test retry decorator when all retries are exhausted."""
        call_count = 0

        caplog.set_level(logging.WARNING, logger="clarity.core.decorators")

        with caplog.at_level(logging.WARNING):

            @retry_on_failure(max_retries=2, delay_seconds=0.01)
            def always_failing_function() -> str:
                nonlocal call_count
                call_count += 1
                msg = f"Failure {call_count}"
                raise ValueError(msg)

            with pytest.raises(ValueError, match="Failure 3"):
                always_failing_function()

        assert call_count == 3  # Initial attempt + 2 retries
        log_messages = " ".join(caplog.messages)
        assert "failed after 3 attempts" in log_messages

    @staticmethod
    async def test_retry_on_failure_async() -> None:
        """Test retry decorator with async functions."""
        call_count = 0

        @retry_on_failure(max_retries=2, delay_seconds=0.01)
        async def async_flaky_function() -> str:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.001)
            if call_count < 2:
                msg = f"Async attempt {call_count} failed"
                raise ValueError(msg)
            return f"async_success_{call_count}"

        result = await async_flaky_function()
        assert result == "async_success_2"
        assert call_count == 2

    @staticmethod
    def test_retry_on_failure_exponential_backoff() -> None:
        """Test retry decorator with exponential backoff."""
        call_times = []

        @retry_on_failure(max_retries=3, delay_seconds=0.01, exponential_backoff=True)
        def backoff_function() -> str:
            call_times.append(time.time())
            if len(call_times) < 3:
                msg = "Not ready yet"
                raise ValueError(msg)
            return "backoff_success"

        result = backoff_function()
        assert result == "backoff_success"
        assert len(call_times) == 3

        # Check that delays increase exponentially
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        assert delay2 > delay1  # Second delay should be longer

    @staticmethod
    def test_retry_on_failure_specific_exceptions() -> None:
        """Test retry decorator with specific exception types."""
        call_count = 0

        @retry_on_failure(max_retries=2, delay_seconds=0.01, exceptions=(ValueError,))
        def selective_retry_function() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                msg = "Retryable error"
                raise ValueError(msg)
            if call_count == 2:
                msg = "Non-retryable error"
                raise TypeError(msg)
            return "should_not_reach"

        with pytest.raises(TypeError, match="Non-retryable error"):
            selective_retry_function()

        assert call_count == 2  # ValueError was retried, TypeError was not


class TestValidateInput:
    """Test validate_input decorator."""

    @staticmethod
    def test_validate_input_valid_case() -> None:
        """Test input validation with valid input."""

        def is_positive(args_kwargs: tuple[tuple[Any, ...], dict[str, Any]]) -> bool:
            args, _ = args_kwargs
            return len(args) > 0 and isinstance(args[0], int) and args[0] > 0

        @validate_input(is_positive, "Input must be a positive integer")
        def process_positive(x: int) -> str:
            return f"processed_{x}"

        result = process_positive(5)
        assert result == "processed_5"

    @staticmethod
    def test_validate_input_invalid_case() -> None:
        """Test input validation with invalid input."""

        def is_positive(args_kwargs: tuple[tuple[Any, ...], dict[str, Any]]) -> bool:
            args, _ = args_kwargs
            return len(args) > 0 and isinstance(args[0], int) and args[0] > 0

        @validate_input(is_positive, "Input must be a positive integer")
        def process_positive(x: int) -> str:
            return f"processed_{x}"

        with pytest.raises(ValueError, match="Input must be a positive integer"):
            process_positive(-1)

        with pytest.raises(ValueError, match="Input must be a positive integer"):
            process_positive(0)

    @staticmethod
    def test_validate_input_with_kwargs() -> None:
        """Test input validation considering kwargs."""

        def has_required_kwargs(
            args_kwargs: tuple[tuple[Any, ...], dict[str, Any]],
        ) -> bool:
            _, kwargs = args_kwargs
            return "required_param" in kwargs and kwargs["required_param"] is not None

        @validate_input(has_required_kwargs, "Missing required_param")
        def function_with_required_kwarg(**kwargs: Any) -> str:
            return f"processed_{kwargs['required_param']}"

        result = function_with_required_kwarg(required_param="test")
        assert result == "processed_test"

        with pytest.raises(ValueError, match="Missing required_param"):
            function_with_required_kwarg(other_param="test")


class TestAuditTrail:
    """Test audit_trail decorator."""

    @staticmethod
    def test_audit_trail_basic(caplog: pytest.LogCaptureFixture) -> None:
        """Test basic audit trail functionality."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @audit_trail("test_operation")
            def audited_function() -> str:
                return "audit_result"

            result = audited_function()

        assert result == "audit_result"
        log_messages = " ".join(caplog.messages)
        assert "test_operation" in log_messages
        assert "status" in log_messages
        assert "success" in log_messages

    @staticmethod
    def test_audit_trail_with_user_and_resource(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test audit trail with user and resource IDs."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @audit_trail(
                "data_access", user_id_param="user_id", resource_id_param="resource_id"
            )
            def data_access_function(user_id: str, resource_id: str) -> str:
                return f"data_for_{user_id}_{resource_id}"

            result = data_access_function(user_id="user123", resource_id="res456")

        assert result == "data_for_user123_res456"
        log_messages = " ".join(caplog.messages)
        assert "user123" in log_messages
        assert "res456" in log_messages
        assert "data_access" in log_messages

    @staticmethod
    def test_audit_trail_exception_handling(caplog: pytest.LogCaptureFixture) -> None:
        """Test audit trail when function raises exception."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @audit_trail("failing_operation")
            def failing_audited_function() -> str:
                msg = "Audit test error"
                raise RuntimeError(msg)

            with pytest.raises(RuntimeError, match="Audit test error"):
                failing_audited_function()

        log_messages = " ".join(caplog.messages)
        assert "failing_operation" in log_messages
        assert "failed" in log_messages
        assert "Audit test error" in log_messages

    @staticmethod
    async def test_audit_trail_async(caplog: pytest.LogCaptureFixture) -> None:
        """Test audit trail with async functions."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @audit_trail("async_operation")
            async def async_audited_function() -> str:
                await asyncio.sleep(0.01)
                return "async_audit_result"

            result = await async_audited_function()

        assert result == "async_audit_result"
        log_messages = " ".join(caplog.messages)
        assert "async_operation" in log_messages
        assert "success" in log_messages


class TestServiceMethod:
    """Test service_method composite decorator."""

    @staticmethod
    def test_service_method_basic(caplog: pytest.LogCaptureFixture) -> None:
        """Test service_method decorator basic functionality."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @service_method()
            def service_function() -> str:
                time.sleep(0.02)  # Small delay for timing
                return "service_result"

            result = service_function()

        assert result == "service_result"
        log_messages = " ".join(caplog.messages)
        # Should have logging from log_execution
        assert "Executing" in log_messages
        assert "Completed" in log_messages

    @staticmethod
    def test_service_method_with_retries(caplog: pytest.LogCaptureFixture) -> None:
        """Test service_method decorator with retry functionality."""
        call_count = 0

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @service_method(max_retries=2)
            def flaky_service_function() -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    msg = "Service temporarily unavailable"
                    raise ValueError(msg)
                return "service_recovered"

            result = flaky_service_function()

        assert result == "service_recovered"
        assert call_count == 2

    @staticmethod
    def test_service_method_timing_threshold(caplog: pytest.LogCaptureFixture) -> None:
        """Test service_method decorator with timing threshold."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @service_method(timing_threshold_ms=10.0)
            def slow_service_function() -> str:
                time.sleep(0.05)  # 50ms delay (above 10ms threshold)
                return "slow_service_result"

            @service_method(timing_threshold_ms=100.0)
            def fast_service_function() -> str:
                time.sleep(0.01)  # 10ms delay (below 100ms threshold)
                return "fast_service_result"

            slow_result = slow_service_function()
            fast_result = fast_service_function()

        assert slow_result == "slow_service_result"
        assert fast_result == "fast_service_result"

        log_messages = " ".join(caplog.messages)
        # Slow function should be logged
        assert "slow_service_function executed in" in log_messages


class TestRepositoryMethod:
    """Test repository_method composite decorator."""

    @staticmethod
    def test_repository_method_basic(caplog: pytest.LogCaptureFixture) -> None:
        """Test repository_method decorator basic functionality."""
        caplog.set_level(logging.DEBUG, logger="clarity.core.decorators")

        with caplog.at_level(logging.DEBUG):

            @repository_method()
            def repository_function() -> str:
                time.sleep(0.02)  # Small delay
                return "repository_result"

            result = repository_function()

        assert result == "repository_result"
        # Should have debug-level logging
        debug_messages = [
            record for record in caplog.records if record.levelno == logging.DEBUG
        ]
        assert len(debug_messages) >= 2  # At least entry and completion

    @staticmethod
    def test_repository_method_with_retries(caplog: pytest.LogCaptureFixture) -> None:
        """Test repository_method decorator with automatic retries."""
        call_count = 0

        caplog.set_level(logging.DEBUG, logger="clarity.core.decorators")

        with caplog.at_level(logging.DEBUG):

            @repository_method()
            def flaky_repository_function() -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    msg = "Database temporarily unavailable"
                    raise ConnectionError(msg)
                return "repository_recovered"

            result = flaky_repository_function()

        assert result == "repository_recovered"
        assert call_count == 3  # Initial + 2 retries (default max_retries=2)

    @staticmethod
    def test_repository_method_timing_threshold(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test repository_method decorator with timing threshold."""
        caplog.set_level(logging.DEBUG, logger="clarity.core.decorators")

        with caplog.at_level(logging.DEBUG):

            @repository_method()
            def slow_repository_function() -> str:
                time.sleep(0.06)  # 60ms delay (above 50ms default threshold)
                return "slow_repository_result"

            result = slow_repository_function()

        assert result == "slow_repository_result"
        log_messages = " ".join(caplog.messages)
        # Should log the timing since it's above threshold
        assert "executed in" in log_messages


class TestDecoratorsIntegration:
    """Test integration and edge cases for decorators."""

    @staticmethod
    def test_multiple_decorators_combination(caplog: pytest.LogCaptureFixture) -> None:
        """Test applying multiple decorators together."""
        call_count = 0

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @log_execution()
            @measure_execution_time()
            @retry_on_failure(max_retries=2, delay_seconds=0.01)
            def multi_decorated_function() -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    msg = "Need retry"
                    raise ValueError(msg)
                time.sleep(0.02)  # For timing measurement
                return "multi_decorated_result"

            result = multi_decorated_function()

        assert result == "multi_decorated_result"
        assert call_count == 2

        log_messages = " ".join(caplog.messages)
        # Should have logging from multiple decorators
        assert "Executing" in log_messages
        assert "Completed" in log_messages
        assert "executed in" in log_messages

    @staticmethod
    def test_decorator_function_metadata_preservation() -> None:
        """Test that decorators preserve function metadata."""

        @service_method()
        def documented_service_function() -> str:
            """This function has documentation and metadata."""
            return "documented_result"

        # Check that function name and docstring are preserved
        assert documented_service_function.__name__ == "documented_service_function"
        assert documented_service_function.__doc__ is not None
        assert "documentation" in documented_service_function.__doc__

    @staticmethod
    async def test_decorator_async_compatibility() -> None:
        """Test that all decorators work correctly with async functions."""
        call_count = 0

        @log_execution()
        @measure_execution_time()
        @retry_on_failure(max_retries=1, delay_seconds=0.01)
        async def comprehensive_async_function() -> str:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            if call_count < 2:
                msg = "Async retry needed"
                raise ValueError(msg)
            return "comprehensive_async_result"

        result = await comprehensive_async_function()
        assert result == "comprehensive_async_result"
        assert call_count == 2

    @staticmethod
    def test_decorator_error_propagation() -> None:
        """Test that decorators properly propagate exceptions."""

        @log_execution()
        @measure_execution_time()
        def error_propagation_function() -> str:
            msg = "Custom test error"
            raise CustomTestError(msg)

        with pytest.raises(CustomTestError, match="Custom test error"):
            error_propagation_function()

    @staticmethod
    def test_decorator_return_type_preservation() -> None:
        """Test that decorators preserve return types."""

        @service_method()
        def typed_function(x: int) -> int:
            return x * 2

        result = typed_function(5)
        assert isinstance(result, int)
        assert result == 10


class CustomTestError(Exception):
    """Custom exception for testing error propagation."""


# Legacy mock tests (keeping for backward compatibility)
class TestDecoratorsBasic:
    """Test basic decorator functionality."""

    @staticmethod
    def test_mock_decorator_behavior() -> None:
        """Test mock decorator behavior for coverage."""

        # Mock a simple decorator pattern
        def simple_decorator(func: F) -> F:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = func(*args, **kwargs)
                return f"decorated_{result}"

            return wrapper  # type: ignore[return-value]

        @simple_decorator
        def test_function() -> str:
            return "original"

        result = test_function()
        assert result == "decorated_original"

    @staticmethod
    def test_async_decorator_pattern() -> None:
        """Test async decorator pattern."""

        def async_decorator(func: F) -> F:
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return f"async_decorated_{result}"

            return wrapper  # type: ignore[return-value]

        @async_decorator
        def sync_function() -> str:
            return "sync"

        @async_decorator
        async def async_function() -> str:
            await asyncio.sleep(0)  # Make it actually async
            return "async"

        # Test sync function
        sync_result: str = asyncio.run(sync_function())  # type: ignore[arg-type]
        assert sync_result == "async_decorated_sync"

        # Test async function
        async_result = asyncio.run(async_function())
        assert async_result == "async_decorated_async"

    @staticmethod
    def test_decorator_with_parameters() -> None:
        """Test decorator with parameters."""

        def parameterized_decorator(prefix: str) -> Callable[[F], F]:
            def decorator(func: F) -> F:
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    result = func(*args, **kwargs)
                    return f"{prefix}_{result}"

                return wrapper  # type: ignore[return-value]

            return decorator

        @parameterized_decorator("TEST")
        def test_function() -> str:
            return "value"

        result = test_function()
        assert result == "TEST_value"

    @staticmethod
    def test_decorator_error_handling() -> None:
        """Test decorator error handling patterns."""

        def error_handling_decorator(func: F) -> F:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return func(*args, **kwargs)
                except ValueError:
                    return "handled_error"
                except Exception as e:  # noqa: BLE001
                    return f"unexpected_error_{type(e).__name__}"

            return wrapper  # type: ignore[return-value]

        @error_handling_decorator
        def function_with_value_error() -> str:
            msg = "Test error"
            raise ValueError(msg)

        @error_handling_decorator
        def function_with_type_error() -> str:
            msg = "Type error"
            raise TypeError(msg)

        @error_handling_decorator
        def normal_function() -> str:
            return "normal"

        assert function_with_value_error() == "handled_error"
        assert function_with_type_error() == "unexpected_error_TypeError"
        assert normal_function() == "normal"

    @staticmethod
    def test_caching_decorator_pattern() -> None:
        """Test caching decorator pattern."""
        cache: dict[str, Any] = {}

        def simple_cache_decorator(func: F) -> F:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Simple cache key from args
                cache_key = f"{func.__name__}_{args!s}_{kwargs!s}"

                if cache_key in cache:
                    return f"cached_{cache[cache_key]}"

                result = func(*args, **kwargs)
                cache[cache_key] = result
                return result

            return wrapper  # type: ignore[return-value]

        call_count = 0

        @simple_cache_decorator
        def expensive_function(x: int) -> str:
            nonlocal call_count
            call_count += 1
            return f"computed_{x}"

        # First call
        result1 = expensive_function(5)
        assert result1 == "computed_5"
        assert call_count == 1

        # Second call (should use cache)
        result2 = expensive_function(5)
        assert result2 == "cached_computed_5"
        assert call_count == 1  # Not incremented

    @staticmethod
    def test_timing_decorator_pattern() -> None:
        """Test timing decorator pattern."""
        times: list[float] = []

        def timing_decorator(func: F) -> F:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                times.append(end_time - start_time)
                return result

            return wrapper  # type: ignore[return-value]

        @timing_decorator
        def timed_function() -> str:
            time.sleep(0.01)  # Small delay
            return "timed"

        result = timed_function()
        assert result == "timed"
        assert len(times) == 1
        assert times[0] > 0.005  # Should take at least some time

    @staticmethod
    def test_validation_decorator_pattern() -> None:
        """Test validation decorator pattern."""

        def validate_positive(func: Callable[[int], str]) -> Callable[[int], str]:
            def wrapper(x: int) -> str:
                if x <= 0:
                    return "invalid_input"
                return func(x)

            return wrapper

        @validate_positive
        def process_positive_number(x: int) -> str:
            return f"processed_{x}"

        assert process_positive_number(5) == "processed_5"
        assert process_positive_number(-1) == "invalid_input"
        assert process_positive_number(0) == "invalid_input"

    @staticmethod
    def test_retry_decorator_pattern() -> None:
        """Test retry decorator pattern."""

        def retry_decorator(max_attempts: int = 3) -> Callable[[F], F]:
            def decorator(func: F) -> F:
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    for attempt in range(max_attempts):
                        try:
                            return func(*args, **kwargs)
                        except Exception:  # noqa: BLE001
                            if attempt == max_attempts - 1:
                                break
                    return f"failed_after_{max_attempts}_attempts"

                return wrapper  # type: ignore[return-value]

            return decorator

        attempt_count = 0

        @retry_decorator(max_attempts=3)
        def flaky_function() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                msg = f"Attempt {attempt_count}"
                raise ValueError(msg)
            return "success"

        result = flaky_function()
        assert result == "success"
        assert attempt_count == 3

    @staticmethod
    def test_decorator_metadata_preservation() -> None:
        """Test that decorators preserve function metadata."""

        def metadata_preserving_decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        @metadata_preserving_decorator
        def documented_function() -> str:
            """This function has documentation."""
            return "documented"

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ is not None
        assert "documentation" in documented_function.__doc__

    @staticmethod
    def test_multiple_decorators() -> None:
        """Test applying multiple decorators."""

        def add_prefix(func: F) -> F:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = func(*args, **kwargs)
                return f"prefix_{result}"

            return wrapper  # type: ignore[return-value]

        def add_suffix(func: F) -> F:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = func(*args, **kwargs)
                return f"{result}_suffix"

            return wrapper  # type: ignore[return-value]

        @add_prefix
        @add_suffix
        def base_function() -> str:
            return "base"

        result = base_function()
        assert result == "prefix_base_suffix"

    @staticmethod
    def test_class_method_decorator() -> None:
        """Test decorator on class methods."""

        def log_method_calls(func: F) -> F:
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                result = func(self, *args, **kwargs)
                return f"logged_{result}"

            return wrapper  # type: ignore[return-value]

        class TestClass:
            """Test class for method decoration."""

            @log_method_calls
            def test_method(self) -> str:
                """Test method."""
                return "method_result"

        instance = TestClass()
        result = instance.test_method()
        assert result == "logged_method_result"

    @staticmethod
    def test_decorator_with_state() -> None:
        """Test decorator that maintains state."""

        def counting_decorator(func: F) -> F:
            call_count = 0

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                nonlocal call_count
                call_count += 1
                result = func(*args, **kwargs)
                wrapper.get_call_count = lambda: call_count  # type: ignore[attr-defined]
                return f"call_{call_count}_{result}"

            return wrapper  # type: ignore[return-value]

        @counting_decorator
        def counted_function() -> str:
            return "result"

        assert counted_function() == "call_1_result"
        assert counted_function() == "call_2_result"
        assert counted_function.get_call_count() == 2  # type: ignore[attr-defined]

    @staticmethod
    async def test_async_decorator_error_handling() -> None:
        """Test async decorator with error handling."""

        def async_error_handler(func: F) -> F:
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    return func(*args, **kwargs)
                except Exception:  # noqa: BLE001 - Test needs to catch all exceptions
                    return "async_error_handled"

            return wrapper  # type: ignore[return-value]

        @async_error_handler
        async def failing_async_function() -> str:
            await asyncio.sleep(0)  # Make it actually async
            async_error_msg = "Async error"
            raise ValueError(async_error_msg)

        @async_error_handler
        def failing_sync_function() -> str:
            sync_error_msg = "Sync error"
            raise ValueError(sync_error_msg)

        async_result = await failing_async_function()
        assert async_result == "async_error_handled"

        # For sync function decorated with async decorator, we need to await it
        sync_result = await failing_sync_function()  # type: ignore[misc]
        assert sync_result == "async_error_handled"
