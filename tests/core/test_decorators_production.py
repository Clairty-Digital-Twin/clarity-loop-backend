"""Comprehensive tests for core decorators - production infrastructure.

Tests critical system infrastructure including logging, timing, retries,
validation, and audit trails that production systems depend on.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import logging
import time
from typing import Any, Never

from _pytest.logging import LogCaptureFixture
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
    """Test logging execution decorator functionality."""

    def test_log_execution_sync_function_basic(self, caplog: LogCaptureFixture) -> None:
        """Test basic logging for synchronous functions."""

        @log_execution()
        def test_function() -> str:
            return "success"

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):
            result = test_function()

        assert result == "success"
        log_messages = [record.message for record in caplog.records]
        assert any("Executing" in msg for msg in log_messages)
        assert any("Completed" in msg for msg in log_messages)

    def test_log_execution_with_args_and_result(
        self, caplog: LogCaptureFixture
    ) -> None:
        """Test logging with arguments and result included."""

        @log_execution(include_args=True, include_result=True)
        def test_function(x: int, y: int, _z: str | None = None) -> int:  # noqa: PT019
            return x + y

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):
            result = test_function(1, 2, _z="test")

        assert result == 3
        log_messages = [record.message for record in caplog.records]
        assert any("args=(1, 2)" in msg for msg in log_messages)
        assert any("kwargs={'_z': 'test'}" in msg for msg in log_messages)

    def test_log_execution_exception_handling(self, caplog: LogCaptureFixture) -> None:
        """Test logging when function raises exception."""

        @log_execution()
        def failing_function() -> Never:
            msg = "Test error"
            raise ValueError(msg)

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with (
            caplog.at_level(logging.INFO),
            pytest.raises(ValueError, match="Test error"),
        ):
            failing_function()

        log_messages = [record.message for record in caplog.records]
        assert any("Executing" in msg for msg in log_messages)
        assert any(record.levelname == "ERROR" for record in caplog.records)

    @pytest.mark.asyncio
    async def test_log_execution_async_function(
        self, caplog: LogCaptureFixture
    ) -> None:
        """Test logging for async functions."""

        @log_execution()
        async def async_test_function() -> str:
            await asyncio.sleep(0.001)
            return "async success"

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):
            result = await async_test_function()

        assert result == "async success"
        log_messages = [record.message for record in caplog.records]
        assert any("Executing" in msg for msg in log_messages)
        assert any("Completed" in msg for msg in log_messages)


class TestMeasureExecutionTimeDecorator:
    """Test execution time measurement decorator."""

    def test_measure_execution_time_basic(self, caplog: LogCaptureFixture) -> None:
        """Test execution time measurement for sync functions."""

        @measure_execution_time()
        def test_function() -> str:
            time.sleep(0.01)  # 10ms delay
            return "timed"

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):
            result = test_function()

        assert result == "timed"
        log_messages = [record.message for record in caplog.records]
        assert any("executed in" in msg and "ms" in msg for msg in log_messages)

    def test_measure_execution_time_with_threshold(
        self, caplog: LogCaptureFixture
    ) -> None:
        """Test execution time measurement with threshold."""

        @measure_execution_time(threshold_ms=50.0)
        def fast_function() -> str:
            return "fast"

        @measure_execution_time(threshold_ms=5.0)
        def slow_function() -> str:
            time.sleep(0.01)  # 10ms delay, above 5ms threshold
            return "slow"

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):
            fast_result = fast_function()
            slow_result = slow_function()

        assert fast_result == "fast"
        assert slow_result == "slow"

        log_messages = [record.message for record in caplog.records]
        # Fast function should not log (below threshold)
        assert not any("fast_function executed in" in msg for msg in log_messages)
        # Slow function should log (above threshold)
        assert any("slow_function executed in" in msg for msg in log_messages)

    def test_measure_execution_time_exception(self, caplog: LogCaptureFixture) -> None:
        """Test execution time measurement when function fails."""

        @measure_execution_time()
        def failing_function() -> Never:
            time.sleep(0.01)
            msg = "Timing test error"
            raise Exception(msg)  # noqa: TRY002

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with (
            caplog.at_level(logging.INFO),
            pytest.raises(Exception, match="Timing test error"),
        ):
            failing_function()

        log_messages = [record.message for record in caplog.records]
        assert any("failed after" in msg and "ms" in msg for msg in log_messages)

    @pytest.mark.asyncio
    async def test_measure_execution_time_async(
        self, caplog: LogCaptureFixture
    ) -> None:
        """Test execution time measurement for async functions."""

        @measure_execution_time()
        async def async_test_function() -> str:
            await asyncio.sleep(0.01)
            return "async timed"

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):
            result = await async_test_function()

        assert result == "async timed"
        log_messages = [record.message for record in caplog.records]
        assert any("executed in" in msg and "ms" in msg for msg in log_messages)


class TestRetryOnFailureDecorator:
    """Test retry on failure decorator functionality."""

    def test_retry_success_first_attempt(self, caplog: LogCaptureFixture) -> None:
        """Test retry decorator when function succeeds on first attempt."""

        @retry_on_failure(max_retries=3)
        def successful_function() -> str:
            return "success"

        caplog.set_level(logging.WARNING, logger="clarity.core.decorators")

        caplog.set_level(logging.WARNING, logger="clarity.core.decorators")

        with caplog.at_level(logging.WARNING):
            result = successful_function()

        assert result == "success"
        # Should not have any retry logs
        assert not any("retrying" in record.message for record in caplog.records)

    def test_retry_success_after_failures(self, caplog: LogCaptureFixture) -> None:
        """Test retry decorator when function succeeds after retries."""
        call_count = 0

        @retry_on_failure(max_retries=2, delay_seconds=0.001)
        def flaky_function() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                msg = "Temporary failure"
                raise ConnectionError(msg)
            return "eventual success"

        caplog.set_level(logging.WARNING, logger="clarity.core.decorators")

        caplog.set_level(logging.WARNING, logger="clarity.core.decorators")

        with caplog.at_level(logging.WARNING):
            result = flaky_function()

        assert result == "eventual success"
        assert call_count == 3
        log_messages = [record.message for record in caplog.records]
        assert any("retrying" in msg for msg in log_messages)

    def test_retry_exceeds_max_retries(self, caplog: LogCaptureFixture) -> None:
        """Test retry decorator when max retries are exceeded."""
        call_count = 0

        @retry_on_failure(max_retries=2, delay_seconds=0.001)
        def always_failing_function() -> Never:
            nonlocal call_count
            call_count += 1
            msg = "Always fails"
            raise ValueError(msg)

        caplog.set_level(logging.WARNING, logger="clarity.core.decorators")

        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(ValueError, match="Always fails"),
        ):
            always_failing_function()

        assert call_count == 3  # Initial call + 2 retries
        log_messages = [record.message for record in caplog.records]
        assert any("failed after 3 attempts" in msg for msg in log_messages)

    def test_retry_exponential_backoff(self) -> None:
        """Test retry decorator with exponential backoff."""
        call_times = []

        @retry_on_failure(max_retries=2, delay_seconds=0.001, exponential_backoff=True)
        def timing_function() -> Never:
            call_times.append(time.time())
            msg = "Timing test"
            raise Exception(msg)  # noqa: TRY002

        with pytest.raises(Exception, match="Timing test"):
            timing_function()

        assert len(call_times) == 3
        # Check that delays increase
        time_diff_1 = call_times[1] - call_times[0]
        time_diff_2 = call_times[2] - call_times[1]
        assert time_diff_2 > time_diff_1

    def test_retry_specific_exceptions(self) -> None:
        """Test retry decorator with specific exception types."""

        @retry_on_failure(
            max_retries=2, delay_seconds=0.001, exceptions=(ConnectionError,)
        )
        def connection_error_function() -> Never:
            msg = "Connection failed"
            raise ConnectionError(msg)

        @retry_on_failure(
            max_retries=2, delay_seconds=0.001, exceptions=(ConnectionError,)
        )
        def value_error_function() -> Never:
            msg = "Value error - should not retry"
            raise ValueError(msg)

        # ConnectionError should be retried
        with pytest.raises(ConnectionError):
            connection_error_function()

        # ValueError should not be retried
        with pytest.raises(ValueError, match="Value error - should not retry"):
            value_error_function()

    @pytest.mark.asyncio
    async def test_retry_async_function(self, caplog: LogCaptureFixture) -> None:
        """Test retry decorator with async functions."""
        call_count = 0

        @retry_on_failure(max_retries=2, delay_seconds=0.001)
        async def async_flaky_function() -> str:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0)  # Make it properly async
            if call_count < 3:
                msg = "Async timeout"
                raise TimeoutError(msg)
            return "async success"

        caplog.set_level(logging.WARNING, logger="clarity.core.decorators")

        caplog.set_level(logging.WARNING, logger="clarity.core.decorators")

        with caplog.at_level(logging.WARNING):
            result = await async_flaky_function()

        assert result == "async success"
        assert call_count == 3


class TestValidateInputDecorator:
    """Test input validation decorator."""

    def test_validate_input_success(self) -> None:
        """Test input validation when validation passes."""

        def positive_number_validator(
            args_kwargs: tuple[tuple[Any, ...], dict[str, Any]],
        ) -> bool:
            args, _kwargs = args_kwargs
            return len(args) > 0 and isinstance(args[0], (int, float)) and args[0] > 0

        @validate_input(positive_number_validator, "Number must be positive")
        def process_number(num: float) -> int | float:
            return num * 2

        result = process_number(5)
        assert result == 10

    def test_validate_input_failure(self) -> None:
        """Test input validation when validation fails."""

        def positive_number_validator(
            args_kwargs: tuple[tuple[Any, ...], dict[str, Any]],
        ) -> bool:
            args, _kwargs = args_kwargs
            return len(args) > 0 and isinstance(args[0], (int, float)) and args[0] > 0

        @validate_input(positive_number_validator, "Number must be positive")
        def process_number(num: float) -> int | float:
            return num * 2

        with pytest.raises(ValueError, match="Number must be positive"):
            process_number(-5)

    def test_validate_input_with_kwargs(self) -> None:
        """Test input validation with keyword arguments."""

        def email_validator(
            args_kwargs: tuple[tuple[Any, ...], dict[str, Any]],
        ) -> bool:
            _args, kwargs = args_kwargs
            email = kwargs.get("email", "")
            return "@" in email and "." in email

        @validate_input(email_validator, "Invalid email format")
        def send_email(message: str, email: str | None = None) -> str:
            return f"Sent: {message} to {email}"

        result = send_email("Hello", email="test@example.com")
        assert "test@example.com" in result

        with pytest.raises(ValueError, match="Invalid email format"):
            send_email("Hello", email="invalid")


class TestAuditTrailDecorator:
    """Test audit trail decorator functionality."""

    def test_audit_trail_basic(self, caplog: LogCaptureFixture) -> None:
        """Test basic audit trail functionality."""

        @audit_trail("user_login")
        def login_user() -> dict[str, str]:
            return {"status": "success", "user_id": "123"}

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):
            result = login_user()

        assert result["status"] == "success"
        log_messages = [record.message for record in caplog.records]
        assert any("user_login" in msg for msg in log_messages)
        assert any("success" in msg for msg in log_messages)

    def test_audit_trail_with_user_and_resource_ids(
        self, caplog: LogCaptureFixture
    ) -> None:
        """Test audit trail with user and resource ID extraction."""

        @audit_trail(
            "delete_document", user_id_param="user_id", resource_id_param="doc_id"
        )
        def delete_document(user_id: str, doc_id: str) -> dict[str, bool]:
            _ = user_id  # Used by decorator
            _ = doc_id  # Used by decorator
            return {"deleted": True}

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):
            result = delete_document(user_id="user123", doc_id="doc456")

        assert result["deleted"] is True
        log_messages = [record.message for record in caplog.records]
        assert any("user123" in msg for msg in log_messages)
        assert any("doc456" in msg for msg in log_messages)

    def test_audit_trail_exception(self, caplog: LogCaptureFixture) -> None:
        """Test audit trail when function raises exception."""

        @audit_trail("risky_operation")
        def failing_operation() -> Never:
            msg = "Access denied"
            raise PermissionError(msg)

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with (
            caplog.at_level(logging.INFO),
            pytest.raises(PermissionError, match="Access denied"),
        ):
            failing_operation()

        log_messages = [record.message for record in caplog.records]
        assert any("failed" in msg for msg in log_messages)
        assert any("Access denied" in msg for msg in log_messages)

    @pytest.mark.asyncio
    async def test_audit_trail_async(self, caplog: LogCaptureFixture) -> None:
        """Test audit trail with async functions."""

        @audit_trail("async_operation", user_id_param="user_id")
        async def async_operation(user_id: str) -> dict[str, bool]:
            _ = user_id  # Used by decorator
            await asyncio.sleep(0.001)
            return {"completed": True}

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):
            result = await async_operation(user_id="async_user")

        assert result["completed"] is True
        log_messages = [record.message for record in caplog.records]
        assert any("async_operation" in msg for msg in log_messages)
        assert any("async_user" in msg for msg in log_messages)


class TestCompositeDecorators:
    """Test composite decorators (service_method, repository_method)."""

    def test_service_method_decorator(self, caplog: LogCaptureFixture) -> None:
        """Test service method composite decorator."""

        @service_method(log_level=logging.INFO, timing_threshold_ms=0.0)
        def service_function(data: str) -> str:
            time.sleep(0.001)
            return f"Processed: {data}"

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):
            result = service_function("test data")

        assert result == "Processed: test data"
        log_messages = [record.message for record in caplog.records]
        assert any("Executing" in msg for msg in log_messages)
        assert any("executed in" in msg for msg in log_messages)

    def test_service_method_with_retries(self, caplog: LogCaptureFixture) -> None:
        """Test service method with retry functionality."""
        call_count = 0

        @service_method(max_retries=2, timing_threshold_ms=0.0)
        def unreliable_service(data: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                msg = "Service unavailable"
                raise ConnectionError(msg)
            return f"Service result: {data}"

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):
            result = unreliable_service("retry test")

        assert result == "Service result: retry test"
        assert call_count == 2

    def test_repository_method_decorator(self, caplog: LogCaptureFixture) -> None:
        """Test repository method composite decorator."""

        @repository_method(log_level=logging.DEBUG, timing_threshold_ms=0.0)
        def repository_function(query: str) -> dict[str, Any]:
            time.sleep(0.001)
            return {"results": [1, 2, 3], "query": query}

        caplog.set_level(logging.DEBUG, logger="clarity.core.decorators")

        caplog.set_level(logging.DEBUG, logger="clarity.core.decorators")

        with caplog.at_level(logging.DEBUG):
            result = repository_function("SELECT * FROM users")

        assert len(result["results"]) == 3
        assert result["query"] == "SELECT * FROM users"
        log_messages = [record.message for record in caplog.records]
        assert any("Executing" in msg for msg in log_messages)
        assert any("executed in" in msg for msg in log_messages)

    def test_repository_method_with_retries(self, caplog: LogCaptureFixture) -> None:
        """Test repository method with default retry functionality."""
        call_count = 0

        @repository_method(timing_threshold_ms=0.0)
        def flaky_repository() -> dict[str, str]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                msg = "Database connection lost"
                raise ConnectionError(msg)
            return {"data": "retrieved"}

        caplog.set_level(logging.DEBUG, logger="clarity.core.decorators")

        caplog.set_level(logging.DEBUG, logger="clarity.core.decorators")

        with caplog.at_level(logging.DEBUG):
            result = flaky_repository()

        assert result["data"] == "retrieved"
        assert call_count == 3


class TestProductionScenarios:
    """Test realistic production scenarios."""

    def test_complete_service_layer_stack(self, caplog: LogCaptureFixture) -> None:
        """Test complete service layer with all decorators."""

        @audit_trail(
            "process_user_request",
            user_id_param="user_id",
            resource_id_param="request_id",
        )
        @service_method(log_level=logging.INFO, timing_threshold_ms=1.0, max_retries=1)
        def process_user_request(
            user_id: str, request_id: str, data: str
        ) -> dict[str, str]:
            time.sleep(0.002)  # 2ms to exceed timing threshold
            return {
                "user_id": user_id,
                "request_id": request_id,
                "processed_data": data.upper(),
                "timestamp": datetime.now(UTC).isoformat(),
            }

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        caplog.set_level(logging.INFO, logger="clarity.core.decorators")

        with caplog.at_level(logging.INFO):
            result = process_user_request(
                user_id="user123", request_id="req456", data="important request"
            )

        assert result["user_id"] == "user123"
        assert result["request_id"] == "req456"
        assert result["processed_data"] == "IMPORTANT REQUEST"

        log_messages = [record.message for record in caplog.records]
        assert any("process_user_request" in msg for msg in log_messages)
        assert any("user123" in msg for msg in log_messages)
        assert any("req456" in msg for msg in log_messages)

    def test_database_operation_with_retries(self, caplog: LogCaptureFixture) -> None:
        """Test database operation with retry and timing."""
        connection_attempts = 0

        @repository_method(
            log_level=logging.DEBUG, timing_threshold_ms=0.0, max_retries=3
        )
        def save_user_data(user_data: dict[str, str]) -> dict[str, Any]:
            nonlocal connection_attempts
            connection_attempts += 1

            if connection_attempts <= 2:
                msg = "Database connection failed"
                raise ConnectionError(msg)

            return {"saved": True, "user_id": user_data["id"]}

        user_data = {"id": "user789", "name": "Test User"}

        caplog.set_level(logging.DEBUG, logger="clarity.core.decorators")

        caplog.set_level(logging.DEBUG, logger="clarity.core.decorators")

        with caplog.at_level(logging.DEBUG):
            result = save_user_data(user_data)

        assert result["saved"] is True
        assert result["user_id"] == "user789"
        assert connection_attempts == 3

        log_messages = [record.message for record in caplog.records]
        assert any("retrying" in msg for msg in log_messages)

    def test_input_validation_production_context(self) -> None:
        """Test input validation in realistic production scenario."""

        def validate_api_request(
            args_kwargs: tuple[tuple[Any, ...], dict[str, Any]],
        ) -> bool:
            args, _kwargs = args_kwargs
            if len(args) == 0:
                return False

            request_data = args[0]
            if not isinstance(request_data, dict):
                return False

            required_fields = ["user_id", "action", "data"]
            return all(field in request_data for field in required_fields)

        @validate_input(validate_api_request, "Invalid API request format")
        @audit_trail("api_request", user_id_param="user_id")
        def handle_api_request(
            request_data: dict[str, Any], user_id: str | None = None
        ) -> dict[str, str]:
            _ = user_id  # Used by decorator
            return {
                "status": "processed",
                "action": request_data["action"],
                "user_id": request_data["user_id"],
            }

        valid_request = {
            "user_id": "user123",
            "action": "update_profile",
            "data": {"name": "New Name"},
        }

        result = handle_api_request(valid_request, user_id="user123")
        assert result["status"] == "processed"
        assert result["action"] == "update_profile"

        invalid_request = {"user_id": "user123"}  # Missing required fields
        with pytest.raises(ValueError, match="Invalid API request format"):
            handle_api_request(invalid_request, user_id="user123")
