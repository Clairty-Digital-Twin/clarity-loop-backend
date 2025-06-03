"""Comprehensive tests for decorators.

Tests all decorators and edge cases to improve coverage from 11% to 90%+.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Tuple

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


class TestLogExecutionDecorator:  # ruff: noqa: PLR0904
    """Comprehensive tests for log_execution decorator."""

    def test_log_execution_sync_function_default_params(self, caplog):
        """Test log_execution decorator on sync function with default parameters."""
        with caplog.at_level(logging.INFO):
            @log_execution()
            def test_func(x, y):
                return x + y

            result = test_func(1, 2)

        assert result == 3
        # Check that function execution was logged (with full module path)
        assert "Executing" in caplog.text and "test_func" in caplog.text
        assert "Completed" in caplog.text and "test_func" in caplog.text

    def test_log_execution_sync_function_with_args_result(self, caplog):
        """Test log_execution decorator with include_args and include_result."""
        with caplog.at_level(logging.INFO):
            @log_execution(include_args=True, include_result=True)
            def test_func(x, y=10):
                return x * y

            result = test_func(5, y=3)

        assert result == 15
        assert "with args=(5,), kwargs={'y': 3}" in caplog.text
        assert "-> 15" in caplog.text

    def test_log_execution_sync_function_with_exception(self, caplog):
        """Test log_execution decorator when function raises exception."""
        with caplog.at_level(logging.INFO):
            @log_execution()
            def test_func():
                raise ValueError("Test error")

            with pytest.raises(ValueError, match="Test error"):
                test_func()

        # Check that error was logged (with full module path)
        assert "Error in" in caplog.text and "test_func" in caplog.text

    async def test_log_execution_async_function_default_params(self, caplog):
        """Test log_execution decorator on async function with default parameters."""
        with caplog.at_level(logging.INFO):
            @log_execution()
            async def test_func(x, y):
                await asyncio.sleep(0.001)
                return x + y

            result = await test_func(1, 2)

        assert result == 3
        # Check that async function execution was logged (with full module path)
        assert "Executing" in caplog.text and "test_func" in caplog.text
        assert "Completed" in caplog.text and "test_func" in caplog.text

    async def test_log_execution_async_function_with_args_result(self, caplog):
        """Test log_execution decorator on async function with args and result."""
        with caplog.at_level(logging.DEBUG):
            @log_execution(level=logging.DEBUG, include_args=True, include_result=True)
            async def test_func(name, age=25):
                await asyncio.sleep(0.001)
                return f"{name}-{age}"

            result = await test_func("test", age=30)

        assert result == "test-30"
        assert "with args=('test',), kwargs={'age': 30}" in caplog.text
        assert "-> test-30" in caplog.text

    async def test_log_execution_async_function_with_exception(self, caplog):
        """Test log_execution decorator on async function with exception."""
        with caplog.at_level(logging.INFO):
            @log_execution()
            async def test_func():
                await asyncio.sleep(0.001)
                raise RuntimeError("Async error")

            with pytest.raises(RuntimeError, match="Async error"):
                await test_func()

        # Check that async error was logged (with full module path)
        assert "Error in" in caplog.text and "test_func" in caplog.text

    def test_log_execution_custom_log_level(self, caplog):
        """Test log_execution decorator with custom log level."""
        with caplog.at_level(logging.WARNING):
            @log_execution(level=logging.WARNING)
            def test_func():
                return "warning level"

            result = test_func()

        assert result == "warning level"
        # Check that custom log level function was executed
        assert "Executing" in caplog.text and "test_func" in caplog.text


class TestMeasureExecutionTimeDecorator:
    """Comprehensive tests for measure_execution_time decorator."""

    def test_measure_execution_time_sync_function(self, caplog):
        """Test measure_execution_time decorator on sync function."""
        with caplog.at_level(logging.INFO):
            @measure_execution_time()
            def test_func():
                time.sleep(0.001)  # Small delay
                return "timed"

            result = test_func()

        assert result == "timed"
        assert "test_func executed in" in caplog.text
        assert "ms" in caplog.text

    def test_measure_execution_time_with_threshold(self, caplog):
        """Test measure_execution_time decorator with threshold."""
        with caplog.at_level(logging.INFO):
            @measure_execution_time(threshold_ms=1000.0)  # High threshold
            def test_func():
                return "fast"

            result = test_func()

        assert result == "fast"
        # Should not log because execution time is below threshold
        assert "test_func executed in" not in caplog.text

    def test_measure_execution_time_sync_with_exception(self, caplog):
        """Test measure_execution_time decorator when sync function raises exception."""
        with caplog.at_level(logging.INFO):
            @measure_execution_time()
            def test_func():
                time.sleep(0.001)
                raise ValueError("Timing error")

            with pytest.raises(ValueError, match="Timing error"):
                test_func()

        assert "test_func failed after" in caplog.text
        assert "ms" in caplog.text

    async def test_measure_execution_time_async_function(self, caplog):
        """Test measure_execution_time decorator on async function."""
        with caplog.at_level(logging.INFO):
            @measure_execution_time()
            async def test_func():
                await asyncio.sleep(0.001)
                return "async timed"

            result = await test_func()

        assert result == "async timed"
        assert "test_func executed in" in caplog.text

    async def test_measure_execution_time_async_with_exception(self, caplog):
        """Test measure_execution_time decorator when async function raises exception."""
        with caplog.at_level(logging.INFO):
            @measure_execution_time()
            async def test_func():
                await asyncio.sleep(0.001)
                raise RuntimeError("Async timing error")

            with pytest.raises(RuntimeError, match="Async timing error"):
                await test_func()

        assert "test_func failed after" in caplog.text

    def test_measure_execution_time_custom_log_level(self, caplog):
        """Test measure_execution_time decorator with custom log level."""
        with caplog.at_level(logging.DEBUG):
            @measure_execution_time(log_level=logging.DEBUG)
            def test_func():
                return "debug timing"

            result = test_func()

        assert result == "debug timing"
        assert "test_func executed in" in caplog.text


class TestRetryOnFailureDecorator:
    """Comprehensive tests for retry_on_failure decorator."""

    def test_retry_on_failure_sync_success_first_try(self, caplog):
        """Test retry decorator when sync function succeeds on first try."""
        with caplog.at_level(logging.WARNING):
            @retry_on_failure(max_retries=2)
            def test_func():
                return "success"

            result = test_func()

        assert result == "success"
        # Should not log any retry attempts
        assert "failed (attempt" not in caplog.text

    def test_retry_on_failure_sync_success_after_retries(self, caplog):
        """Test retry decorator when sync function succeeds after retries."""
        call_count = 0

        with caplog.at_level(logging.WARNING):
            @retry_on_failure(max_retries=2, delay_seconds=0.001)
            def test_func():
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ValueError("Retry me")
                return "success after retries"

            result = test_func()

        assert result == "success after retries"
        assert call_count == 3
        assert "failed (attempt 1/3)" in caplog.text
        assert "failed (attempt 2/3)" in caplog.text

    def test_retry_on_failure_sync_max_retries_exceeded(self, caplog):
        """Test retry decorator when sync function fails all retries."""
        with caplog.at_level(logging.WARNING):
            @retry_on_failure(max_retries=1, delay_seconds=0.001)
            def test_func():
                raise ConnectionError("Always fails")

            with pytest.raises(ConnectionError, match="Always fails"):
                test_func()

        assert "failed after 2 attempts" in caplog.text

    def test_retry_on_failure_exponential_backoff(self):
        """Test retry decorator with exponential backoff."""
        call_times = []

        @retry_on_failure(max_retries=2, delay_seconds=0.01, exponential_backoff=True)
        def test_func():
            call_times.append(time.time())
            raise ValueError("Test exponential backoff")

        with pytest.raises(ValueError):
            test_func()

        # Check that delays increased exponentially
        assert len(call_times) == 3
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        assert delay2 > delay1  # Second delay should be longer

    def test_retry_on_failure_linear_backoff(self):
        """Test retry decorator without exponential backoff."""
        call_times = []

        @retry_on_failure(max_retries=2, delay_seconds=0.01, exponential_backoff=False)
        def test_func():
            call_times.append(time.time())
            raise ValueError("Test linear backoff")

        with pytest.raises(ValueError):
            test_func()

        # Check that delays are consistent
        assert len(call_times) == 3
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        # Delays should be approximately equal (within 10ms tolerance)
        assert abs(delay1 - delay2) < 0.01

    def test_retry_on_failure_specific_exceptions(self):
        """Test retry decorator with specific exception types."""
        call_count = 0

        @retry_on_failure(max_retries=2, exceptions=(ValueError, TypeError))
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First error")
            if call_count == 2:
                raise TypeError("Second error")
            return "success"

        result = test_func()
        assert result == "success"
        assert call_count == 3

    def test_retry_on_failure_non_retryable_exception(self):
        """Test retry decorator with non-retryable exception."""
        call_count = 0

        @retry_on_failure(max_retries=2, exceptions=(ValueError,))
        def test_func():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Non-retryable")

        with pytest.raises(RuntimeError, match="Non-retryable"):
            test_func()

        # Should only be called once since RuntimeError is not in exceptions list
        assert call_count == 1

    async def test_retry_on_failure_async_success_after_retries(self, caplog):
        """Test retry decorator on async function with retries."""
        call_count = 0

        with caplog.at_level(logging.WARNING):
            @retry_on_failure(max_retries=1, delay_seconds=0.001)
            async def test_func():
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise ValueError("Async retry me")
                return "async success"

            result = await test_func()

        assert result == "async success"
        assert call_count == 2
        assert "failed (attempt 1/2)" in caplog.text

    async def test_retry_on_failure_async_max_retries_exceeded(self, caplog):
        """Test retry decorator when async function fails all retries."""
        with caplog.at_level(logging.WARNING):
            @retry_on_failure(max_retries=1, delay_seconds=0.001)
            async def test_func():
                raise ConnectionError("Always fails async")

            with pytest.raises(ConnectionError, match="Always fails async"):
                await test_func()

        assert "failed after 2 attempts" in caplog.text


class TestValidateInputDecorator:
    """Comprehensive tests for validate_input decorator."""

    @staticmethod
    def test_validate_input_valid_args() -> None:
        """Test validate_input decorator with valid arguments."""
        def validator(args_kwargs: Tuple[Tuple[Any, ...], Dict[str, Any]]) -> bool:
            args, _ = args_kwargs
            return len(args) > 0 and isinstance(args[0], str)

        @validate_input(validator, "Input must be a non-empty string")
        def test_func(text: str) -> str:
            return f"processed: {text}"

        result = test_func("hello")
        assert result == "processed: hello"

    @staticmethod
    def test_validate_input_invalid_args() -> None:
        """Test validate_input decorator with invalid arguments."""
        def validator(args_kwargs: Tuple[Tuple[Any, ...], Dict[str, Any]]) -> bool:
            args, _ = args_kwargs
            return len(args) > 0 and isinstance(args[0], str)

        @validate_input(validator, "Input must be a non-empty string")
        def test_func(text: Any) -> str:
            return f"processed: {text}"

        with pytest.raises(ValueError, match="Input must be a non-empty string"):
            test_func(123)

    @staticmethod
    def test_validate_input_empty_args() -> None:
        """Test validate_input decorator with empty arguments."""
        def validator(args_kwargs: Tuple[Tuple[Any, ...], Dict[str, Any]]) -> bool:
            args, _ = args_kwargs
            return len(args) > 0

        @validate_input(validator, "At least one argument required")
        def test_func() -> str:
            return "no args"

        with pytest.raises(ValueError, match="At least one argument required"):
            test_func()

    @staticmethod
    def test_validate_input_kwargs_validation() -> None:
        """Test validate_input decorator with keyword arguments validation."""
        def validator(args_kwargs: Tuple[Tuple[Any, ...], Dict[str, Any]]) -> bool:
            _, kwargs = args_kwargs
            return "required_key" in kwargs

        @validate_input(validator, "Missing required keyword argument")
        def test_func(**kwargs: Any) -> str:
            return f"got: {kwargs}"

        result = test_func(required_key="value")
        assert "value" in result

        with pytest.raises(ValueError, match="Missing required keyword argument"):
            test_func(other_key="value")

    @staticmethod
    def test_validate_input_complex_validation() -> None:
        """Test validate_input decorator with complex validation logic."""
        def validator(args_kwargs: Tuple[Tuple[Any, ...], Dict[str, Any]]) -> bool:
            args, _kwargs = args_kwargs
            if len(args) == 0:
                return False
            data = args[0]
            return isinstance(data, dict) and "id" in data and "name" in data

        @validate_input(validator, "Data must be dict with id and name")
        def test_func(data: Dict[str, Any]) -> str:
            return data['id'] + data['name']

        result = test_func({"id": "123", "name": "test"})
        assert result == "123test"

        with pytest.raises(ValueError, match="Data must be dict with id and name"):
            test_func({"id": "123"})  # Missing name


class TestAuditTrailDecorator:
    """Comprehensive tests for audit_trail decorator."""

    @staticmethod
    def test_audit_trail_sync_function_success(caplog: pytest.LogCaptureFixture) -> None:
        """Test audit_trail decorator on successful sync function."""
        with caplog.at_level(logging.INFO):
            @audit_trail("test_operation")
            def test_func() -> str:
                return "audited result"

            result = test_func()

        assert result == "audited result"
        assert "Audit started:" in caplog.text
        assert "test_operation" in caplog.text
        assert "Audit completed:" in caplog.text

    @staticmethod
    def test_audit_trail_sync_function_with_user_resource_ids(caplog: pytest.LogCaptureFixture) -> None:
        """Test audit_trail decorator with user_id and resource_id parameters."""
        with caplog.at_level(logging.INFO):
            @audit_trail("update_resource", user_id_param="user", resource_id_param="resource")
            def test_func(_user: Any = None, _resource: Any = None) -> str:
                return "updated"

            result = test_func(_user="user123", _resource="resource456")

        assert result == "updated"
        assert "user123" in caplog.text
        assert "resource456" in caplog.text

    @staticmethod
    def test_audit_trail_sync_function_failure(caplog: pytest.LogCaptureFixture) -> None:
        """Test audit_trail decorator when sync function fails."""
        with caplog.at_level(logging.INFO):
            @audit_trail("failing_operation")
            def test_func() -> None:
                msg = "Audit this error"
                raise ValueError(msg)

            with pytest.raises(ValueError, match="Audit this error"):
                test_func()

        assert "Audit started:" in caplog.text
        assert "failing_operation" in caplog.text
        assert "Audit failed:" in caplog.text

    @staticmethod
    async def test_audit_trail_async_function_success(caplog: pytest.LogCaptureFixture) -> None:
        """Test audit_trail decorator on successful async function."""
        with caplog.at_level(logging.INFO):
            @audit_trail("async_operation")
            async def test_func() -> str:
                await asyncio.sleep(0.001)
                return "async audited"

            result = await test_func()

        assert result == "async audited"
        assert "Audit started:" in caplog.text
        assert "async_operation" in caplog.text
        assert "success" in caplog.text

    @staticmethod
    async def test_audit_trail_async_function_failure(caplog: pytest.LogCaptureFixture) -> None:
        """Test audit_trail decorator when async function fails."""
        with caplog.at_level(logging.INFO):
            @audit_trail("async_failing_operation")
            async def test_func() -> None:
                await asyncio.sleep(0.001)
                msg = "Async audit error"
                raise RuntimeError(msg)

            with pytest.raises(RuntimeError, match="Async audit error"):
                await test_func()

        assert "Audit started:" in caplog.text
        assert "Async audit error" in caplog.text

    @staticmethod
    def test_audit_trail_missing_user_resource_params(caplog: pytest.LogCaptureFixture) -> None:
        """Test audit_trail decorator when user/resource params are not provided."""
        with caplog.at_level(logging.INFO):
            @audit_trail("test_op", user_id_param="missing_user", resource_id_param="missing_resource")
            def test_func() -> str:
                return "no params"

            result = test_func()

        assert result == "no params"
        assert "Audit started:" in caplog.text


class TestServiceMethodDecorator:
    """Comprehensive tests for service_method composite decorator."""

    @staticmethod
    def test_service_method_default_configuration(caplog: pytest.LogCaptureFixture) -> None:
        """Test service_method decorator with default configuration."""
        with caplog.at_level(logging.INFO):
            @service_method()
            def test_func() -> str:
                time.sleep(0.001)
                return "service result"

            result = test_func()

        assert result == "service result"
        assert "Executing" in caplog.text
        assert "Completed" in caplog.text
        # Timing may or may not be logged depending on execution time and thresholds

    @staticmethod
    def test_service_method_custom_configuration(caplog: pytest.LogCaptureFixture) -> None:
        """Test service_method decorator with custom configuration."""
        with caplog.at_level(logging.DEBUG):
            @service_method(log_level=logging.DEBUG, timing_threshold_ms=0.1, max_retries=1)
            def test_func() -> str:
                return "custom service"

            result = test_func()

        assert result == "custom service"
        assert "Executing" in caplog.text

    @staticmethod
    def test_service_method_with_retries(caplog: pytest.LogCaptureFixture) -> None:
        """Test service_method decorator with retry functionality."""
        call_count = 0

        with caplog.at_level(logging.WARNING):
            @service_method(max_retries=1, log_level=logging.WARNING)
            def test_func() -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    msg = "Service retry"
                    raise ValueError(msg)
                return "service success"

            result = test_func()

        assert result == "service success"
        assert call_count == 2
        assert "failed (attempt" in caplog.text

    @staticmethod
    async def test_service_method_async_function(caplog: pytest.LogCaptureFixture) -> None:
        """Test service_method decorator on async function."""
        with caplog.at_level(logging.INFO):
            @service_method()
            async def test_func() -> str:
                await asyncio.sleep(0.001)
                return "async service"

            result = await test_func()

        assert result == "async service"
        assert "Executing" in caplog.text


class TestRepositoryMethodDecorator:
    """Comprehensive tests for repository_method composite decorator."""

    @staticmethod
    def test_repository_method_default_configuration(caplog: pytest.LogCaptureFixture) -> None:
        """Test repository_method decorator with default configuration."""
        with caplog.at_level(logging.DEBUG):
            @repository_method()
            def test_func() -> str:
                time.sleep(0.001)
                return "repository result"

            result = test_func()

        assert result == "repository result"
        assert "Executing" in caplog.text
        assert "Completed" in caplog.text

    @staticmethod
    def test_repository_method_with_retries(caplog: pytest.LogCaptureFixture) -> None:
        """Test repository_method decorator with default retry functionality."""
        call_count = 0

        with caplog.at_level(logging.WARNING):
            @repository_method(log_level=logging.WARNING)
            def test_func() -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    msg = "Repository connection failed"
                    raise ConnectionError(msg)
                return "repository success"

            result = test_func()

        assert result == "repository success"
        assert call_count == 2
        assert "failed (attempt" in caplog.text

    @staticmethod
    def test_repository_method_custom_configuration(caplog: pytest.LogCaptureFixture) -> None:
        """Test repository_method decorator with custom configuration."""
        with caplog.at_level(logging.INFO):
            @repository_method(log_level=logging.INFO, timing_threshold_ms=0.1, max_retries=0)
            def test_func() -> str:
                return "custom repository"

            result = test_func()

        assert result == "custom repository"
        assert "Executing" in caplog.text

    @staticmethod
    async def test_repository_method_async_function(caplog: pytest.LogCaptureFixture) -> None:
        """Test repository_method decorator on async function."""
        with caplog.at_level(logging.DEBUG):
            @repository_method()
            async def test_func():
                await asyncio.sleep(0.001)
                return "async repository"

            result = await test_func()

        assert result == "async repository"
        assert "Executing" in caplog.text


class TestDecoratorEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_decorator_on_method_with_self(self, caplog):
        """Test decorators work correctly with class methods."""

        class TestClass:
            @log_execution()
            def method(self, x):
                return x * 2

            @measure_execution_time()
            async def async_method(self, x):
                await asyncio.sleep(0.001)
                return x + 1

        with caplog.at_level(logging.INFO):
            obj = TestClass()
            result = obj.method(5)
            async_result = asyncio.run(obj.async_method(10))

        assert result == 10
        assert async_result == 11
        assert "TestClass.method" in caplog.text
        assert "TestClass.async_method" in caplog.text

    def test_decorator_preserves_function_metadata(self):
        """Test that decorators preserve function metadata."""
        @log_execution()
        @measure_execution_time()
        def documented_function(x, y):
            """This function adds two numbers."""
            return x + y

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ is not None
        assert "adds two numbers" in documented_function.__doc__

    def test_multiple_decorators_composition(self, caplog):
        """Test multiple decorators working together."""
        call_count = 0

        with caplog.at_level(logging.INFO):
            @audit_trail("complex_operation")
            @retry_on_failure(max_retries=1, delay_seconds=0.001)
            @measure_execution_time()
            @log_execution(include_args=True, include_result=True)
            def complex_func(value):
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise ValueError("First attempt fails")
                return value * 3

            result = complex_func(7)

        assert result == 21
        assert call_count == 2
        # Check that all decorators logged appropriately
        assert "Executing" in caplog.text
        assert "executed in" in caplog.text
        assert "failed (attempt" in caplog.text
        assert "Audit:" in caplog.text

    def test_decorator_with_zero_retries(self):
        """Test retry decorator with zero retries (should not retry)."""
        call_count = 0

        @retry_on_failure(max_retries=0)
        def test_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("No retries")

        with pytest.raises(ValueError, match="No retries"):
            test_func()

        assert call_count == 1  # Should only be called once

    def test_timing_threshold_edge_case(self, caplog):
        """Test timing decorator at threshold boundary."""
        with caplog.at_level(logging.INFO):
            @measure_execution_time(threshold_ms=0.0)  # Log everything
            def test_func():
                return "threshold test"

            result = test_func()

        assert result == "threshold test"
        assert "executed in" in caplog.text

    def test_audit_trail_with_complex_data_types(self, caplog):
        """Test audit_trail decorator with complex parameter types."""
        with caplog.at_level(logging.INFO):
            @audit_trail("complex_data_op", user_id_param="user_data")
            def test_func(user_data=None):
                return "processed"

            # Test with dict parameter
            result = test_func(user_data={"id": "user123", "name": "Test User"})

        assert result == "processed"
        # Should handle complex data types in audit log
        assert "Audit:" in caplog.text

    def test_validate_input_with_none_args(self):
        """Test validate_input decorator with None arguments."""
        def validator(args_kwargs):
            args, kwargs = args_kwargs
            return args is not None and kwargs is not None

        @validate_input(validator, "Args and kwargs cannot be None")
        def test_func():
            return "validated"

        result = test_func()
        assert result == "validated"
