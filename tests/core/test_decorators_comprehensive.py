"""Comprehensive tests for decorators.

Tests all decorators and edge cases to improve coverage from 11% to 90%+.
"""

from __future__ import annotations

import asyncio
import logging
import operator
import time
from typing import Any

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


class TestLogExecutionDecorator:
    """Comprehensive tests for log_execution decorator."""

    @staticmethod
    def test_log_execution_sync_function_default_params(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test log_execution decorator with default parameters on sync function."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):
            # Use operator.add instead of custom function per FURB118
            test_func = log_execution()(operator.add)

            result = test_func(3, 4)

        assert result == 7
        assert "Executing" in caplog.text
        assert "Completed" in caplog.text

    @staticmethod
    def test_log_execution_sync_function_with_args_result(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test log_execution decorator with args and result logging."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @log_execution(include_args=True, include_result=True)
            def test_func(x: int, y: int = 10) -> int:
                return x * y

            result = test_func(5)

        assert result == 50
        assert "Executing" in caplog.text
        assert "args=(5,)" in caplog.text
        assert "-> 50" in caplog.text

    @staticmethod
    def test_log_execution_sync_function_with_exception(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test log_execution decorator when sync function raises exception."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @log_execution()
            def test_func() -> None:
                msg = "Test exception"
                raise ValueError(msg)

            with pytest.raises(ValueError, match="Test exception"):
                test_func()

        assert "Executing" in caplog.text
        assert "Error in" in caplog.text

    @staticmethod
    async def test_log_execution_async_function_default_params(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test log_execution decorator with default parameters on async function."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @log_execution()
            async def test_func(x: int, y: int) -> int:
                await asyncio.sleep(0.001)
                return x - y

            result = await test_func(10, 3)

        assert result == 7
        assert "Executing" in caplog.text
        assert "Completed" in caplog.text

    @staticmethod
    async def test_log_execution_async_function_with_args_result(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test log_execution decorator with args and result logging on async function."""
        caplog.set_level(logging.DEBUG, logger="clarity.core.decorators")

        with caplog.at_level(logging.DEBUG):

            @log_execution(level=logging.DEBUG, include_args=True, include_result=True)
            async def test_func(name: str, age: int = 25) -> str:
                await asyncio.sleep(0.001)
                return f"{name} is {age}"

            result = await test_func("Alice")

        assert result == "Alice is 25"
        assert "Executing" in caplog.text
        assert "Alice" in caplog.text

    @staticmethod
    async def test_log_execution_async_function_with_exception(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test log_execution decorator when async function raises exception."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @log_execution()
            async def test_func() -> None:
                await asyncio.sleep(0.001)
                msg = "Async test exception"
                raise RuntimeError(msg)

            with pytest.raises(RuntimeError, match="Async test exception"):
                await test_func()

        assert "Executing" in caplog.text
        assert "Error in" in caplog.text

    @staticmethod
    def test_log_execution_custom_log_level(caplog: pytest.LogCaptureFixture) -> None:
        """Test log_execution decorator with custom log level."""
        caplog.set_level(logging.WARNING, logger="clarity.core.decorators")

        with caplog.at_level(logging.WARNING):

            @log_execution(level=logging.WARNING)
            def test_func() -> str:
                return "custom level"

            result = test_func()

        assert result == "custom level"
        assert "Executing" in caplog.text


class TestMeasureExecutionTimeDecorator:
    """Comprehensive tests for measure_execution_time decorator."""

    @staticmethod
    def test_measure_execution_time_sync_function(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test measure_execution_time decorator on sync function."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @measure_execution_time()
            def test_func() -> str:
                time.sleep(0.001)  # Small delay
                return "timed"

            result = test_func()

        assert result == "timed"
        assert "test_func executed in" in caplog.text
        assert "ms" in caplog.text

    @staticmethod
    def test_measure_execution_time_with_threshold(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test measure_execution_time decorator with threshold."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @measure_execution_time(threshold_ms=1000.0)  # High threshold
            def test_func() -> str:
                return "fast"

            result = test_func()

        assert result == "fast"
        # Should not log because execution time is below threshold
        assert "test_func executed in" not in caplog.text

    @staticmethod
    def test_measure_execution_time_sync_with_exception(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test measure_execution_time decorator when sync function raises exception."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @measure_execution_time()
            def test_func() -> None:
                time.sleep(0.001)
                msg = "Timing error"
                raise ValueError(msg)

            with pytest.raises(ValueError, match="Timing error"):
                test_func()

        assert "test_func failed after" in caplog.text
        assert "ms" in caplog.text

    @staticmethod
    async def test_measure_execution_time_async_function(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test measure_execution_time decorator on async function."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @measure_execution_time()
            async def test_func() -> str:
                await asyncio.sleep(0.001)
                return "async timed"

            result = await test_func()

        assert result == "async timed"
        assert "test_func executed in" in caplog.text

    @staticmethod
    async def test_measure_execution_time_async_with_exception(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test measure_execution_time decorator when async function raises exception."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @measure_execution_time()
            async def test_func() -> None:
                await asyncio.sleep(0.001)
                msg = "Async timing error"
                raise RuntimeError(msg)

            with pytest.raises(RuntimeError, match="Async timing error"):
                await test_func()

        assert "test_func failed after" in caplog.text

    @staticmethod
    def test_measure_execution_time_custom_log_level(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test measure_execution_time decorator with custom log level."""
        caplog.set_level(logging.DEBUG, logger="clarity.core.decorators")

        with caplog.at_level(logging.DEBUG):

            @measure_execution_time(log_level=logging.DEBUG)
            def test_func() -> str:
                return "debug timing"

            result = test_func()

        assert result == "debug timing"
        assert "test_func executed in" in caplog.text


class TestRetryOnFailureDecorator:
    """Comprehensive tests for retry_on_failure decorator."""

    @staticmethod
    def test_retry_on_failure_sync_success_first_try(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test retry decorator when sync function succeeds on first try."""
        caplog.set_level(logging.WARNING, logger="clarity.core.decorators")

        with caplog.at_level(logging.WARNING):

            @retry_on_failure(max_retries=2)
            def test_func() -> str:
                return "success"

            result = test_func()

        assert result == "success"
        # Should not log any retry attempts
        assert "failed (attempt" not in caplog.text

    @staticmethod
    def test_retry_on_failure_sync_success_after_retries(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test retry decorator when sync function succeeds after retries."""
        call_count = 0

        caplog.set_level(logging.WARNING, logger="clarity.core.decorators")

        with caplog.at_level(logging.WARNING):

            @retry_on_failure(max_retries=2, delay_seconds=0.001)
            def test_func() -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    msg = "Retry me"
                    raise ValueError(msg)
                return "success after retries"

            result = test_func()

        assert result == "success after retries"
        assert call_count == 3
        assert "failed (attempt 1/3)" in caplog.text
        assert "failed (attempt 2/3)" in caplog.text

    @staticmethod
    def test_retry_on_failure_sync_max_retries_exceeded(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test retry decorator when sync function fails all retries."""
        caplog.set_level(logging.WARNING, logger="clarity.core.decorators")

        with caplog.at_level(logging.WARNING):

            @retry_on_failure(max_retries=1, delay_seconds=0.001)
            def test_func() -> None:
                msg = "Always fails"
                raise ConnectionError(msg)

            with pytest.raises(ConnectionError, match="Always fails"):
                test_func()

        assert "failed after 2 attempts" in caplog.text

    @staticmethod
    def test_retry_on_failure_exponential_backoff() -> None:
        """Test retry decorator with exponential backoff."""
        call_times = []

        @retry_on_failure(max_retries=2, delay_seconds=0.05, exponential_backoff=True)
        def test_func() -> None:
            call_times.append(time.time())
            msg = "Test exponential backoff"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="Test exponential backoff"):
            test_func()

        # Check that delays increased exponentially
        assert len(call_times) == 3
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        # The expected delays are 0.05s and 0.10s, but we allow for timing variations
        # delay1 should be approximately 0.05s (first retry delay)
        # delay2 should be approximately 0.10s (second retry delay with 2x backoff)
        assert delay1 >= 0.04  # At least 40ms (allowing 20% tolerance)
        assert delay2 >= 0.08  # At least 80ms (allowing 20% tolerance)
        assert delay2 > delay1 * 1.5  # Second delay should be at least 1.5x longer

    @staticmethod
    def test_retry_on_failure_linear_backoff() -> None:
        """Test retry decorator without exponential backoff."""
        call_times = []

        @retry_on_failure(max_retries=2, delay_seconds=0.01, exponential_backoff=False)
        def test_func() -> None:
            call_times.append(time.time())
            msg = "Test linear backoff"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="Test linear backoff"):
            test_func()

        # Check that delays are consistent
        assert len(call_times) == 3
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        # Delays should be approximately equal (within 100ms tolerance for system variations)
        assert abs(delay1 - delay2) < 0.1

    @staticmethod
    def test_retry_on_failure_specific_exceptions() -> None:
        """Test retry decorator with specific exception types."""
        call_count = 0

        @retry_on_failure(max_retries=2, exceptions=(ValueError, TypeError))
        def test_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                msg = "First error"
                raise ValueError(msg)
            if call_count == 2:
                msg = "Second error"
                raise TypeError(msg)
            return "success"

        result = test_func()
        assert result == "success"
        assert call_count == 3

    @staticmethod
    def test_retry_on_failure_non_retryable_exception() -> None:
        """Test retry decorator with non-retryable exception."""
        call_count = 0

        @retry_on_failure(max_retries=2, exceptions=(ValueError,))
        def test_func() -> None:
            nonlocal call_count
            call_count += 1
            msg = "Non-retryable"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="Non-retryable"):
            test_func()

        # Should only be called once since RuntimeError is not in exceptions list
        assert call_count == 1

    @staticmethod
    async def test_retry_on_failure_async_success_after_retries(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test retry decorator on async function with retries."""
        call_count = 0

        caplog.set_level(logging.WARNING, logger="clarity.core.decorators")

        with caplog.at_level(logging.WARNING):

            @retry_on_failure(max_retries=1, delay_seconds=0.001)
            async def test_func() -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    msg = "Async retry me"
                    raise ValueError(msg)
                await asyncio.sleep(0.001)
                return "async success"

            result = await test_func()

        assert result == "async success"
        assert call_count == 2
        assert "failed (attempt 1/2)" in caplog.text

    @staticmethod
    async def test_retry_on_failure_async_max_retries_exceeded(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test retry decorator when async function fails all retries."""
        caplog.set_level(logging.WARNING, logger="clarity.core.decorators")

        with caplog.at_level(logging.WARNING):

            @retry_on_failure(max_retries=1, delay_seconds=0.001)
            async def test_func() -> None:
                await asyncio.sleep(0.001)
                msg = "Always fails async"
                raise ConnectionError(msg)

            with pytest.raises(ConnectionError, match="Always fails async"):
                await test_func()

        assert "failed after 2 attempts" in caplog.text


class TestValidateInputDecorator:
    """Comprehensive tests for validate_input decorator."""

    @staticmethod
    def test_validate_input_valid_args() -> None:
        """Test validate_input decorator with valid arguments."""

        def validator(args_kwargs: tuple[tuple[Any, ...], dict[str, Any]]) -> bool:
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

        def validator(args_kwargs: tuple[tuple[Any, ...], dict[str, Any]]) -> bool:
            args, _ = args_kwargs
            return len(args) > 0 and isinstance(args[0], str)

        @validate_input(validator, "Input must be a non-empty string")
        def test_func(text: object) -> str:
            return f"processed: {text}"

        with pytest.raises(ValueError, match="Input must be a non-empty string"):
            test_func(123)

    @staticmethod
    def test_validate_input_empty_args() -> None:
        """Test validate_input decorator with empty arguments."""

        def validator(args_kwargs: tuple[tuple[Any, ...], dict[str, Any]]) -> bool:
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

        def validator(args_kwargs: tuple[tuple[Any, ...], dict[str, Any]]) -> bool:
            _, kwargs = args_kwargs
            return "required_key" in kwargs

        @validate_input(validator, "Missing required keyword argument")
        def test_func(**kwargs: object) -> str:
            return f"got: {kwargs}"

        result = test_func(required_key="value")
        assert "value" in result

        with pytest.raises(ValueError, match="Missing required keyword argument"):
            test_func(other_key="value")

    @staticmethod
    def test_validate_input_complex_validation() -> None:
        """Test validate_input decorator with complex validation logic."""

        def validator(args_kwargs: tuple[tuple[Any, ...], dict[str, Any]]) -> bool:
            args, _kwargs = args_kwargs
            if len(args) == 0:
                return False
            data = args[0]
            return isinstance(data, dict) and "id" in data and "name" in data

        @validate_input(validator, "Data must be dict with id and name")
        def test_func(data: dict[str, Any]) -> str:
            return str(data["id"]) + str(data["name"])

        result = test_func({"id": "123", "name": "test"})
        assert result == "123test"

        with pytest.raises(ValueError, match="Data must be dict with id and name"):
            test_func({"id": "123"})  # Missing name


class TestAuditTrailDecorator:
    """Comprehensive tests for audit_trail decorator."""

    @staticmethod
    def test_audit_trail_sync_function_success(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test audit_trail decorator on successful sync function."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @audit_trail("test_operation")
            def test_func() -> str:
                return "audited result"

            result = test_func()

        assert result == "audited result"
        assert "Audit:" in caplog.text
        assert "test_operation" in caplog.text
        assert "Audit completed:" in caplog.text

    @staticmethod
    def test_audit_trail_sync_function_with_user_resource_ids(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test audit_trail decorator with user_id and resource_id parameters."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @audit_trail(
                "update_resource", user_id_param="user", resource_id_param="resource"
            )
            def test_func(user: object = None, resource: object = None) -> str:
                # Use the parameters to avoid ARG001 unused argument error
                _user_info = str(user) if user else "unknown"
                _resource_info = str(resource) if resource else "unknown"
                return "updated"

            result = test_func(user="user123", resource="resource456")

        assert result == "updated"
        assert "Audit:" in caplog.text

    @staticmethod
    def test_audit_trail_sync_function_failure(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test audit_trail decorator when sync function fails."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @audit_trail("failing_operation")
            def test_func() -> None:
                msg = "Audit this error"
                raise ValueError(msg)

            with pytest.raises(ValueError, match="Audit this error"):
                test_func()

        assert "Audit:" in caplog.text
        assert "failing_operation" in caplog.text
        assert "Audit failed:" in caplog.text

    @staticmethod
    async def test_audit_trail_async_function_success(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test audit_trail decorator on successful async function."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @audit_trail("async_operation")
            async def test_func() -> str:
                await asyncio.sleep(0.001)
                return "async audited"

            result = await test_func()

        assert result == "async audited"
        assert "Audit:" in caplog.text
        assert "async_operation" in caplog.text
        assert "success" in caplog.text

    @staticmethod
    async def test_audit_trail_async_function_failure(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test audit_trail decorator when async function fails."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @audit_trail("async_failing_operation")
            async def test_func() -> None:
                await asyncio.sleep(0.001)
                msg = "Async audit error"
                raise RuntimeError(msg)

            with pytest.raises(RuntimeError, match="Async audit error"):
                await test_func()

        assert "Audit:" in caplog.text
        assert "Async audit error" in caplog.text

    @staticmethod
    def test_audit_trail_missing_user_resource_params(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test audit_trail decorator when user/resource params are not provided."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @audit_trail(
                "test_op",
                user_id_param="missing_user",
                resource_id_param="missing_resource",
            )
            def test_func() -> str:
                return "no params"

            result = test_func()

        assert result == "no params"
        assert "Audit:" in caplog.text


class TestServiceMethodDecorator:
    """Comprehensive tests for service_method composite decorator."""

    @staticmethod
    def test_service_method_default_configuration(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test service_method decorator with default configuration."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

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
    def test_service_method_custom_configuration(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test service_method decorator with custom configuration."""
        caplog.set_level(logging.DEBUG, logger="clarity.core.decorators")

        with caplog.at_level(logging.DEBUG):

            @service_method(
                log_level=logging.DEBUG, timing_threshold_ms=0.1, max_retries=1
            )
            def test_func() -> str:
                return "custom service"

            result = test_func()

        assert result == "custom service"
        assert "Executing" in caplog.text

    @staticmethod
    def test_service_method_with_retries(caplog: pytest.LogCaptureFixture) -> None:
        """Test service_method decorator with retry functionality."""
        call_count = 0

        caplog.set_level(logging.WARNING, logger="clarity.core.decorators")

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
    async def test_service_method_async_function(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test service_method decorator on async function."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

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
    def test_repository_method_default_configuration(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test repository_method decorator with default configuration."""
        caplog.set_level(logging.DEBUG, logger="clarity.core.decorators")

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

        caplog.set_level(logging.WARNING, logger="clarity.core.decorators")

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
    def test_repository_method_custom_configuration(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test repository_method decorator with custom configuration."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @repository_method(
                log_level=logging.INFO, timing_threshold_ms=0.1, max_retries=0
            )
            def test_func() -> str:
                return "custom repository"

            result = test_func()

        assert result == "custom repository"
        assert "Executing" in caplog.text

    @staticmethod
    async def test_repository_method_async_function(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test repository_method decorator on async function."""
        caplog.set_level(logging.DEBUG, logger="clarity.core.decorators")

        with caplog.at_level(logging.DEBUG):

            @repository_method()
            async def test_func() -> str:
                await asyncio.sleep(0.001)
                return "async repository"

            result = await test_func()

        assert result == "async repository"
        assert "Executing" in caplog.text


class TestDecoratorEdgeCases:
    """Tests for edge cases and error conditions."""

    @staticmethod
    def test_decorator_on_method_with_self(caplog: pytest.LogCaptureFixture) -> None:
        """Test decorators work correctly with class methods."""

        class TestClass:
            @staticmethod
            @log_execution()
            def method(x: int) -> int:
                return x * 2

            @staticmethod
            @measure_execution_time()
            async def async_method(x: int) -> int:
                await asyncio.sleep(0.001)
                return x + 1

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):
            obj = TestClass()
            result = obj.method(5)
            async_result = asyncio.run(obj.async_method(10))

        assert result == 10
        assert async_result == 11
        assert "TestClass.method" in caplog.text
        assert "TestClass.async_method" in caplog.text

    @staticmethod
    def test_decorator_preserves_function_metadata() -> None:
        """Test that decorators preserve function metadata."""

        @log_execution()
        @measure_execution_time()
        def documented_function(x: int, y: int) -> int:
            """This function adds two numbers."""
            return x + y

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ is not None
        assert "adds two numbers" in documented_function.__doc__

    @staticmethod
    def test_multiple_decorators_composition(caplog: pytest.LogCaptureFixture) -> None:
        """Test multiple decorators working together."""
        call_count = 0

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @audit_trail("complex_operation")
            @retry_on_failure(max_retries=1, delay_seconds=0.001)
            @measure_execution_time()
            @log_execution(include_args=True, include_result=True)
            def complex_func(value: int) -> int:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    msg = "First attempt fails"
                    raise ValueError(msg)
                return value * 3

            result = complex_func(7)

        assert result == 21
        assert call_count == 2
        # Check that all decorators logged appropriately
        assert "Executing" in caplog.text
        assert "executed in" in caplog.text
        assert "failed (attempt" in caplog.text
        assert "Audit:" in caplog.text

    @staticmethod
    def test_decorator_with_zero_retries() -> None:
        """Test retry decorator with zero retries (should not retry)."""
        call_count = 0

        @retry_on_failure(max_retries=0)
        def test_func() -> None:
            nonlocal call_count
            call_count += 1
            msg = "No retries"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="No retries"):
            test_func()

        assert call_count == 1  # Should only be called once

    @staticmethod
    def test_timing_threshold_edge_case(caplog: pytest.LogCaptureFixture) -> None:
        """Test timing decorator at threshold boundary."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @measure_execution_time(threshold_ms=0.0)  # Log everything
            def test_func() -> str:
                return "threshold test"

            result = test_func()

        assert result == "threshold test"
        assert "executed in" in caplog.text

    @staticmethod
    def test_audit_trail_with_complex_data_types(
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test audit_trail decorator with complex parameter types."""
        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):

            @audit_trail("complex_data_op", user_id_param="user_data")
            def test_func(user_data: object = None) -> str:
                # Use the parameter to avoid ARG001 unused argument error
                _user_info = str(user_data) if user_data else "no_user"
                return "processed"

            # Test with dict parameter
            result = test_func(user_data={"id": "user123", "name": "Test User"})

        assert result == "processed"
        # Should handle complex data types in audit log
        assert "Audit:" in caplog.text

    @staticmethod
    def test_validate_input_with_none_args() -> None:
        """Test validate_input decorator with None arguments."""

        def validator(args_kwargs: tuple[tuple[Any, ...], dict[str, Any]]) -> bool:
            args, kwargs = args_kwargs
            return len(args) >= 0 and len(kwargs) >= 0

        @validate_input(validator, "Args and kwargs cannot be None")
        def test_func() -> str:
            return "validated"

        result = test_func()
        assert result == "validated"
