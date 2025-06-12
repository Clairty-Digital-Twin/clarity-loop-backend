"""Comprehensive tests for exceptions.

Tests all exception classes and utility functions to improve coverage from 42% to 90%+.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import UUID

from fastapi import Request
from fastapi.responses import JSONResponse

from clarity.core.exceptions import (
    AccessDeniedError,
    AccountDisabledError,
    # Authentication/Authorization exceptions
    AuthenticationError,
    AuthenticationProblem,
    AuthorizationError,
    AuthorizationProblem,
    # Cache exceptions
    CacheError,
    CacheKeyError,
    ClarityAPIException,
    # Base exceptions
    ClarityBaseError,
    # Configuration exceptions
    ConfigurationError,
    ConflictProblem,
    DataLengthMismatchError,
    # Data validation exceptions
    DataValidationError,
    EmptyDataError,
    InferenceError,
    InferenceTimeoutError,
    IntegrationError,
    InternalServerProblem,
    InvalidConfigurationError,
    InvalidNHANESStatsDataError,
    InvalidNHANESStatsError,
    InvalidStepCountDataError,
    MissingConfigurationError,
    # ML model exceptions
    ModelError,
    ModelNotInitializedError,
    NHANESDataNotFoundError,
    # NHANES statistics exceptions
    NHANESStatsError,
    # RFC 7807 classes
    ProblemDetail,
    ProcessingError,
    RateLimitProblem,
    ResourceNotFoundProblem,
    # Service exceptions
    ServiceError,
    ServiceNotInitializedError,
    ServiceUnavailableError,
    ServiceUnavailableProblem,
    ValidationProblem,
    create_numeric_validation_error,
    # Utility functions
    create_validation_error,
    generic_exception_handler,
    # Exception handlers
    problem_detail_exception_handler,
)

if TYPE_CHECKING:
    import pytest


class TestProblemDetail:
    """Tests for RFC 7807 ProblemDetail model."""

    @staticmethod
    def test_problem_detail_creation() -> None:
        """Test creating ProblemDetail with all fields."""
        problem = ProblemDetail(
            type="https://api.clarity.health/problems/test",
            title="Test Problem",
            status=400,
            detail="Test detail message",
            instance="https://api.clarity.health/requests/test-instance",
            trace_id="test-trace-id",
            errors=[{"field": "test", "message": "Test error"}],
            help_url="https://docs.clarity.health/test",
        )

        assert problem.type == "https://api.clarity.health/problems/test"
        assert problem.title == "Test Problem"
        assert problem.status == 400
        assert problem.detail == "Test detail message"
        assert problem.instance == "https://api.clarity.health/requests/test-instance"
        assert problem.trace_id == "test-trace-id"
        assert problem.errors == [{"field": "test", "message": "Test error"}]
        assert problem.help_url == "https://docs.clarity.health/test"

    @staticmethod
    def test_problem_detail_minimal() -> None:
        """Test creating ProblemDetail with minimal required fields."""
        problem = ProblemDetail(
            type="https://api.clarity.health/problems/minimal",
            title="Minimal Problem",
            status=500,
            detail="Minimal detail",
            instance="https://api.clarity.health/requests/minimal",
            trace_id=None,
            errors=None,
            help_url=None,
        )

        assert problem.type == "https://api.clarity.health/problems/minimal"
        assert problem.title == "Minimal Problem"
        assert problem.status == 500
        assert problem.detail == "Minimal detail"
        assert problem.instance == "https://api.clarity.health/requests/minimal"
        assert problem.trace_id is None
        assert problem.errors is None
        assert problem.help_url is None


class TestClarityAPIException:
    """Tests for base ClarityAPIException."""

    @staticmethod
    def test_clarity_api_exception_creation() -> None:
        """Test creating ClarityAPIException with all parameters."""
        exc = ClarityAPIException(
            status_code=400,
            problem_type="https://api.clarity.health/problems/test",
            title="Test Exception",
            detail="Test exception detail",
            instance="test-instance",
            trace_id="test-trace",
            errors=[{"field": "test"}],
            help_url="https://docs.clarity.health/test",
            headers={"X-Test": "header"},
        )

        assert exc.status_code == 400
        assert exc.problem_type == "https://api.clarity.health/problems/test"
        assert exc.title == "Test Exception"
        assert exc.detail == "Test exception detail"
        assert exc.instance == "test-instance"
        assert exc.trace_id == "test-trace"
        assert exc.errors == [{"field": "test"}]
        assert exc.help_url == "https://docs.clarity.health/test"
        assert exc.headers == {"X-Test": "header"}

    @staticmethod
    def test_clarity_api_exception_defaults() -> None:
        """Test ClarityAPIException with default values."""
        exc = ClarityAPIException(
            status_code=500,
            problem_type="https://api.clarity.health/problems/server",
            title="Server Error",
            detail="Internal server error",
        )

        # Instance and trace_id should be auto-generated
        assert exc.instance.startswith("https://api.clarity.health/requests/")
        assert UUID(exc.trace_id)  # Should be valid UUID
        assert exc.errors is None
        assert exc.help_url is None

    @staticmethod
    def test_to_problem_detail() -> None:
        """Test converting ClarityAPIException to ProblemDetail."""
        exc = ClarityAPIException(
            status_code=400,
            problem_type="https://api.clarity.health/problems/test",
            title="Test Exception",
            detail="Test detail",
            trace_id="test-trace-123",
        )

        problem = exc.to_problem_detail()

        assert isinstance(problem, ProblemDetail)
        assert problem.type == exc.problem_type
        assert problem.title == exc.title
        assert problem.status == exc.status_code
        assert problem.detail == exc.detail
        assert problem.instance == exc.instance
        assert problem.trace_id == exc.trace_id


class TestPreDefinedProblemTypes:
    """Tests for pre-defined problem type exceptions."""

    @staticmethod
    def test_validation_problem() -> None:
        """Test ValidationProblem exception."""
        errors = [{"field": "email", "message": "Invalid email format"}]
        exc = ValidationProblem(
            detail="Validation failed",
            errors=errors,
            trace_id="validation-trace",
        )

        assert exc.status_code == 400
        assert (
            exc.problem_type == "https://api.clarity.health/problems/validation-error"
        )
        assert exc.title == "Validation Error"
        assert exc.detail == "Validation failed"
        assert exc.errors == errors
        assert exc.trace_id == "validation-trace"
        assert exc.help_url == "https://docs.clarity.health/errors/validation"

    @staticmethod
    def test_authentication_problem() -> None:
        """Test AuthenticationProblem exception."""
        exc = AuthenticationProblem(detail="Invalid token", trace_id="auth-trace")

        assert exc.status_code == 401
        assert (
            exc.problem_type
            == "https://api.clarity.health/problems/authentication-required"
        )
        assert exc.title == "Authentication Required"
        assert exc.detail == "Invalid token"
        assert exc.trace_id == "auth-trace"
        assert exc.help_url == "https://docs.clarity.health/authentication"

    @staticmethod
    def test_authentication_problem_default() -> None:
        """Test AuthenticationProblem with default message."""
        exc = AuthenticationProblem()

        assert exc.detail == "Authentication required"

    @staticmethod
    def test_authorization_problem() -> None:
        """Test AuthorizationProblem exception."""
        exc = AuthorizationProblem(
            detail="Access denied to resource",
            trace_id="authz-trace",
        )

        assert exc.status_code == 403
        assert (
            exc.problem_type
            == "https://api.clarity.health/problems/authorization-denied"
        )
        assert exc.title == "Authorization Denied"
        assert exc.detail == "Access denied to resource"
        assert exc.trace_id == "authz-trace"

    @staticmethod
    def test_authorization_problem_default() -> None:
        """Test AuthorizationProblem with default message."""
        exc = AuthorizationProblem()

        assert exc.detail == "Insufficient permissions for this resource"

    @staticmethod
    def test_resource_not_found_problem() -> None:
        """Test ResourceNotFoundProblem exception."""
        exc = ResourceNotFoundProblem(
            resource_type="User",
            resource_id="user123",
            trace_id="notfound-trace",
        )

        assert exc.status_code == 404
        assert (
            exc.problem_type == "https://api.clarity.health/problems/resource-not-found"
        )
        assert exc.title == "Resource Not Found"
        assert exc.detail == "User with ID 'user123' does not exist"
        assert exc.trace_id == "notfound-trace"

    @staticmethod
    def test_conflict_problem() -> None:
        """Test ConflictProblem exception."""
        exc = ConflictProblem(
            detail="Email already exists",
            trace_id="conflict-trace",
        )

        assert exc.status_code == 409
        assert (
            exc.problem_type == "https://api.clarity.health/problems/resource-conflict"
        )
        assert exc.title == "Resource Conflict"
        assert exc.detail == "Email already exists"
        assert exc.trace_id == "conflict-trace"

    @staticmethod
    def test_rate_limit_problem() -> None:
        """Test RateLimitProblem exception."""
        exc = RateLimitProblem(
            retry_after=60,
            detail="Too many requests",
            trace_id="ratelimit-trace",
        )

        assert exc.status_code == 429
        assert (
            exc.problem_type
            == "https://api.clarity.health/problems/rate-limit-exceeded"
        )
        assert exc.title == "Rate Limit Exceeded"
        assert exc.detail == "Too many requests"
        assert exc.headers == {"Retry-After": "60"}
        assert exc.trace_id == "ratelimit-trace"

    @staticmethod
    def test_rate_limit_problem_default() -> None:
        """Test RateLimitProblem with default message."""
        exc = RateLimitProblem(retry_after=30)

        assert exc.detail == "Rate limit exceeded"
        assert exc.headers == {"Retry-After": "30"}

    @staticmethod
    def test_internal_server_problem() -> None:
        """Test InternalServerProblem exception."""
        exc = InternalServerProblem(
            detail="Database connection failed",
            trace_id="server-trace",
        )

        assert exc.status_code == 500
        assert (
            exc.problem_type
            == "https://api.clarity.health/problems/internal-server-error"
        )
        assert exc.title == "Internal Server Error"
        assert exc.detail == "Database connection failed"
        assert exc.trace_id == "server-trace"

    @staticmethod
    def test_internal_server_problem_default() -> None:
        """Test InternalServerProblem with default message."""
        exc = InternalServerProblem()

        assert exc.detail == "An internal server error occurred"

    @staticmethod
    def test_service_unavailable_problem() -> None:
        """Test ServiceUnavailableProblem exception."""
        exc = ServiceUnavailableProblem(
            service_name="Database",
            retry_after=120,
            trace_id="unavailable-trace",
        )

        assert exc.status_code == 503
        assert (
            exc.problem_type
            == "https://api.clarity.health/problems/service-unavailable"
        )
        assert exc.title == "Service Unavailable"
        assert exc.detail == "Database is temporarily unavailable"
        assert exc.headers == {"Retry-After": "120"}
        assert exc.trace_id == "unavailable-trace"

    @staticmethod
    def test_service_unavailable_problem_no_retry_after() -> None:
        """Test ServiceUnavailableProblem without retry_after."""
        exc = ServiceUnavailableProblem(service_name="Redis")

        assert exc.detail == "Redis is temporarily unavailable"
        assert exc.headers is None


class TestExceptionHandlers:
    """Tests for exception handler functions."""

    @staticmethod
    def test_problem_detail_exception_handler() -> None:
        """Test problem_detail_exception_handler function."""
        exc = ValidationProblem(
            detail="Test validation error",
            errors=[{"field": "test"}],
        )
        request = MagicMock(spec=Request)

        response = problem_detail_exception_handler(request, exc)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        # FastAPI automatically adds content headers
        assert "Retry-After" not in response.headers

        # Check response content structure
        if isinstance(response.body, bytes):
            content = response.body.decode()
        else:
            content = bytes(response.body).decode()
        assert "validation-error" in content
        assert "Test validation error" in content

    @staticmethod
    def test_problem_detail_exception_handler_with_headers() -> None:
        """Test problem_detail_exception_handler with custom headers."""
        exc = RateLimitProblem(retry_after=60, detail="Rate limited")
        request = MagicMock(spec=Request)

        response = problem_detail_exception_handler(request, exc)

        assert response.status_code == 429
        # Check that our custom header is present (case-insensitive)
        assert "retry-after" in [k.lower() for k in response.headers]
        retry_after_value = response.headers.get("retry-after") or response.headers.get(
            "Retry-After"
        )
        assert retry_after_value == "60"

    @staticmethod
    def test_generic_exception_handler(caplog: pytest.LogCaptureFixture) -> None:
        """Test generic_exception_handler function."""
        exc = ValueError("Unexpected error")
        request = MagicMock(spec=Request)

        response = generic_exception_handler(request, exc)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 500

        # Check that error was logged
        assert "Unhandled exception" in caplog.text
        assert "Unexpected error" in caplog.text

        # Check response content structure
        if isinstance(response.body, bytes):
            content = response.body.decode()
        else:
            content = bytes(response.body).decode()
        assert "internal-server-error" in content
        assert "unexpected error occurred" in content


class TestClarityBaseError:
    """Tests for ClarityBaseError base class."""

    @staticmethod
    def test_clarity_base_error_basic() -> None:
        """Test basic ClarityBaseError creation."""
        exc = ClarityBaseError("Test error message")

        assert str(exc) == "Test error message"
        assert exc.error_code is None
        assert exc.details == {}

    @staticmethod
    def test_clarity_base_error_with_code_and_details() -> None:
        """Test ClarityBaseError with error code and details."""
        details = {"field": "test", "value": 123}
        exc = ClarityBaseError(
            "Test error with code",
            error_code="TEST_ERROR",
            details=details,
        )

        assert str(exc) == "[TEST_ERROR] Test error with code"
        assert exc.error_code == "TEST_ERROR"
        assert exc.details == details

    @staticmethod
    def test_clarity_base_error_empty_details() -> None:
        """Test ClarityBaseError with None details."""
        exc = ClarityBaseError("Test error", details=None)

        assert exc.details == {}


class TestDataValidationExceptions:
    """Tests for data validation exception classes."""

    @staticmethod
    def test_data_validation_error() -> None:
        """Test DataValidationError exception."""
        exc = DataValidationError(
            "Invalid data format",
            field_name="email",
            details={"expected": "email", "got": "string"},
        )

        assert str(exc) == "[DATA_VALIDATION_ERROR] Invalid data format"
        assert exc.error_code == "DATA_VALIDATION_ERROR"
        assert exc.field_name == "email"
        assert exc.details == {"expected": "email", "got": "string"}

    @staticmethod
    def test_data_validation_error_no_field() -> None:
        """Test DataValidationError without field name."""
        exc = DataValidationError("General validation error")

        assert exc.field_name is None

    @staticmethod
    def test_invalid_step_count_data_error() -> None:
        """Test InvalidStepCountDataError exception."""
        exc = InvalidStepCountDataError("Step count must be positive")

        assert str(exc) == "[DATA_VALIDATION_ERROR] Step count must be positive"
        assert isinstance(exc, DataValidationError)

    @staticmethod
    def test_invalid_nhanes_stats_error() -> None:
        """Test InvalidNHANESStatsError exception."""
        exc = InvalidNHANESStatsError("Invalid NHANES format")

        assert str(exc) == "[DATA_VALIDATION_ERROR] Invalid NHANES format"
        assert isinstance(exc, DataValidationError)

    @staticmethod
    def test_processing_error() -> None:
        """Test ProcessingError exception."""
        exc = ProcessingError("Data processing failed")

        assert str(exc) == "[DATA_VALIDATION_ERROR] Data processing failed"
        assert isinstance(exc, DataValidationError)

    @staticmethod
    def test_integration_error() -> None:
        """Test IntegrationError exception."""
        exc = IntegrationError("External API failed")

        assert str(exc) == "[INTEGRATION_ERROR] External API failed"
        assert exc.error_code == "INTEGRATION_ERROR"

    @staticmethod
    def test_data_length_mismatch_error() -> None:
        """Test DataLengthMismatchError exception."""
        exc = DataLengthMismatchError(
            expected_length=10,
            actual_length=5,
            data_type="step_counts",
        )

        expected_msg = "step_counts length mismatch: expected 10, got 5"
        assert str(exc) == f"[DATA_VALIDATION_ERROR] {expected_msg}"
        assert exc.expected_length == 10
        assert exc.actual_length == 5
        assert exc.data_type == "step_counts"

    @staticmethod
    def test_data_length_mismatch_error_default_type() -> None:
        """Test DataLengthMismatchError with default data type."""
        exc = DataLengthMismatchError(expected_length=3, actual_length=7)

        assert "data length mismatch: expected 3, got 7" in str(exc)
        assert exc.data_type == "data"

    @staticmethod
    def test_empty_data_error() -> None:
        """Test EmptyDataError exception."""
        exc = EmptyDataError("measurements")

        assert str(exc) == "[DATA_VALIDATION_ERROR] measurements cannot be empty"
        assert exc.data_type == "measurements"

    @staticmethod
    def test_empty_data_error_default_type() -> None:
        """Test EmptyDataError with default data type."""
        exc = EmptyDataError()

        assert "data cannot be empty" in str(exc)
        assert exc.data_type == "data"


class TestMLModelExceptions:
    """Tests for ML model and inference exception classes."""

    @staticmethod
    def test_model_error() -> None:
        """Test base ModelError exception."""
        exc = ModelError("Model operation failed")

        assert str(exc) == "Model operation failed"
        assert isinstance(exc, ClarityBaseError)

    @staticmethod
    def test_model_not_initialized_error() -> None:
        """Test ModelNotInitializedError exception."""
        exc = ModelNotInitializedError("TransformerModel")

        assert str(exc) == "[MODEL_NOT_INITIALIZED] TransformerModel is not initialized"
        assert exc.model_name == "TransformerModel"
        assert exc.error_code == "MODEL_NOT_INITIALIZED"

    @staticmethod
    def test_model_not_initialized_error_default() -> None:
        """Test ModelNotInitializedError with default model name."""
        exc = ModelNotInitializedError()

        assert "Model is not initialized" in str(exc)
        assert exc.model_name == "Model"

    @staticmethod
    def test_inference_error() -> None:
        """Test InferenceError exception."""
        exc = InferenceError(
            "Inference failed",
            request_id="req123",
            details={"error": "timeout"},
        )

        assert str(exc) == "[INFERENCE_ERROR] Inference failed"
        assert exc.request_id == "req123"
        assert exc.error_code == "INFERENCE_ERROR"
        assert exc.details == {"error": "timeout"}

    @staticmethod
    def test_inference_error_no_request_id() -> None:
        """Test InferenceError without request ID."""
        exc = InferenceError("General inference error")

        assert exc.request_id is None

    @staticmethod
    def test_inference_timeout_error() -> None:
        """Test InferenceTimeoutError exception."""
        exc = InferenceTimeoutError(request_id="req456", timeout_seconds=30.5)

        expected_msg = "Inference request req456 timed out after 30.5s"
        assert str(exc) == f"[INFERENCE_ERROR] {expected_msg}"
        assert exc.request_id == "req456"
        assert exc.timeout_seconds == 30.5
        assert isinstance(exc, InferenceError)


class TestNHANESStatsExceptions:
    """Tests for NHANES statistics exception classes."""

    @staticmethod
    def test_nhanes_stats_error() -> None:
        """Test base NHANESStatsError exception."""
        exc = NHANESStatsError("NHANES operation failed")

        assert str(exc) == "[NHANES_STATS_ERROR] NHANES operation failed"
        assert exc.error_code == "NHANES_STATS_ERROR"

    @staticmethod
    def test_nhanes_data_not_found_error_all_params() -> None:
        """Test NHANESDataNotFoundError with all parameters."""
        exc = NHANESDataNotFoundError(year=2020, age_group="18-25", sex="M")

        expected_msg = "NHANES data not found for year=2020, age_group=18-25, sex=M"
        assert str(exc) == f"[NHANES_STATS_ERROR] {expected_msg}"
        assert exc.year == 2020
        assert exc.age_group == "18-25"
        assert exc.sex == "M"

    @staticmethod
    def test_nhanes_data_not_found_error_partial_params() -> None:
        """Test NHANESDataNotFoundError with partial parameters."""
        exc = NHANESDataNotFoundError(year=2019, sex="F")

        expected_msg = "NHANES data not found for year=2019, sex=F"
        assert str(exc) == f"[NHANES_STATS_ERROR] {expected_msg}"
        assert exc.year == 2019
        assert exc.age_group is None
        assert exc.sex == "F"

    @staticmethod
    def test_nhanes_data_not_found_error_no_params() -> None:
        """Test NHANESDataNotFoundError with no parameters."""
        exc = NHANESDataNotFoundError()

        expected_msg = "NHANES data not found for specified criteria"
        assert str(exc) == f"[NHANES_STATS_ERROR] {expected_msg}"
        assert exc.year is None
        assert exc.age_group is None
        assert exc.sex is None

    @staticmethod
    def test_invalid_nhanes_stats_data_error() -> None:
        """Test InvalidNHANESStatsDataError exception."""
        exc = InvalidNHANESStatsDataError(
            data_type="blood_pressure",
            expected_type="dict",
            actual_type="list",
        )

        expected_msg = "Invalid NHANES blood_pressure: expected dict, got list"
        assert str(exc) == f"[NHANES_STATS_ERROR] {expected_msg}"
        assert exc.data_type == "blood_pressure"
        assert exc.expected_type == "dict"
        assert exc.actual_type == "list"


class TestServiceExceptions:
    """Tests for service exception classes."""

    @staticmethod
    def test_service_error() -> None:
        """Test base ServiceError exception."""
        exc = ServiceError("Service operation failed")

        assert str(exc) == "Service operation failed"
        assert isinstance(exc, ClarityBaseError)

    @staticmethod
    def test_service_not_initialized_error() -> None:
        """Test ServiceNotInitializedError exception."""
        exc = ServiceNotInitializedError("DatabaseService")

        expected_msg = "DatabaseService service is not initialized"
        assert str(exc) == f"[SERVICE_NOT_INITIALIZED] {expected_msg}"
        assert exc.service_name == "DatabaseService"
        assert exc.error_code == "SERVICE_NOT_INITIALIZED"

    @staticmethod
    def test_service_unavailable_error_with_reason() -> None:
        """Test ServiceUnavailableError with reason."""
        exc = ServiceUnavailableError("APIService", reason="Network timeout")

        expected_msg = "APIService service is unavailable: Network timeout"
        assert str(exc) == f"[SERVICE_UNAVAILABLE] {expected_msg}"
        assert exc.service_name == "APIService"
        assert exc.reason == "Network timeout"

    @staticmethod
    def test_service_unavailable_error_no_reason() -> None:
        """Test ServiceUnavailableError without reason."""
        exc = ServiceUnavailableError("CacheService")

        expected_msg = "CacheService service is unavailable"
        assert str(exc) == f"[SERVICE_UNAVAILABLE] {expected_msg}"
        assert exc.service_name == "CacheService"
        assert exc.reason is None


class TestAuthenticationAuthorizationExceptions:
    """Tests for authentication and authorization exception classes."""

    @staticmethod
    def test_authentication_error() -> None:
        """Test base AuthenticationError exception."""
        exc = AuthenticationError("Auth failed")

        assert str(exc) == "[AUTHENTICATION_ERROR] Auth failed"
        assert exc.error_code == "AUTHENTICATION_ERROR"

    @staticmethod
    def test_authorization_error() -> None:
        """Test base AuthorizationError exception."""
        exc = AuthorizationError("Access denied")

        assert str(exc) == "[AUTHORIZATION_ERROR] Access denied"
        assert exc.error_code == "AUTHORIZATION_ERROR"

    @staticmethod
    def test_account_disabled_error() -> None:
        """Test AccountDisabledError exception."""
        exc = AccountDisabledError("user123")

        expected_msg = "User account user123 is disabled"
        assert str(exc) == f"[AUTHENTICATION_ERROR] {expected_msg}"
        assert exc.user_id == "user123"
        assert isinstance(exc, AuthenticationError)

    @staticmethod
    def test_access_denied_error_with_user() -> None:
        """Test AccessDeniedError with user ID."""
        exc = AccessDeniedError("admin_panel", user_id="user456")

        expected_msg = "Access denied to admin_panel for user user456"
        assert str(exc) == f"[AUTHORIZATION_ERROR] {expected_msg}"
        assert exc.resource == "admin_panel"
        assert exc.user_id == "user456"

    @staticmethod
    def test_access_denied_error_no_user() -> None:
        """Test AccessDeniedError without user ID."""
        exc = AccessDeniedError("private_data")

        expected_msg = "Access denied to private_data"
        assert str(exc) == f"[AUTHORIZATION_ERROR] {expected_msg}"
        assert exc.resource == "private_data"
        assert exc.user_id is None


class TestConfigurationExceptions:
    """Tests for configuration exception classes."""

    @staticmethod
    def test_configuration_error() -> None:
        """Test base ConfigurationError exception."""
        exc = ConfigurationError("Config error", config_key="database.url")

        assert str(exc) == "[CONFIGURATION_ERROR] Config error"
        assert exc.config_key == "database.url"

    @staticmethod
    def test_missing_configuration_error() -> None:
        """Test MissingConfigurationError exception."""
        exc = MissingConfigurationError("API_KEY")

        expected_msg = "Missing required configuration: API_KEY"
        assert str(exc) == f"[CONFIGURATION_ERROR] {expected_msg}"
        assert exc.config_key == "API_KEY"

    @staticmethod
    def test_invalid_configuration_error_with_reason() -> None:
        """Test InvalidConfigurationError with reason."""
        exc = InvalidConfigurationError(
            config_key="PORT",
            value="invalid_port",
            reason="must be integer",
        )

        expected_msg = (
            "Invalid configuration value for PORT: invalid_port (must be integer)"
        )
        assert str(exc) == f"[CONFIGURATION_ERROR] {expected_msg}"
        assert exc.config_key == "PORT"
        assert exc.value == "invalid_port"
        assert exc.reason == "must be integer"

    @staticmethod
    def test_invalid_configuration_error_no_reason() -> None:
        """Test InvalidConfigurationError without reason."""
        exc = InvalidConfigurationError(config_key="DEBUG", value=123)

        expected_msg = "Invalid configuration value for DEBUG: 123"
        assert str(exc) == f"[CONFIGURATION_ERROR] {expected_msg}"
        assert exc.config_key == "DEBUG"
        assert exc.value == 123
        assert exc.reason is None


class TestCacheExceptions:
    """Tests for cache exception classes."""

    @staticmethod
    def test_cache_error() -> None:
        """Test base CacheError exception."""
        exc = CacheError("Cache operation failed")

        assert str(exc) == "[CACHE_ERROR] Cache operation failed"
        assert exc.error_code == "CACHE_ERROR"

    @staticmethod
    def test_cache_key_error() -> None:
        """Test CacheKeyError exception."""
        exc = CacheKeyError("user:123", "get")

        expected_msg = "Cache get failed for key: user:123"
        assert str(exc) == f"[CACHE_ERROR] {expected_msg}"
        assert exc.cache_key == "user:123"
        assert exc.operation == "get"


class TestUtilityFunctions:
    """Tests for utility functions."""

    @staticmethod
    def test_create_validation_error() -> None:
        """Test create_validation_error utility function."""
        exc = create_validation_error("age", "int", "not_a_number")

        expected_msg = "Field 'age' expected int, got str: not_a_number"
        assert str(exc) == f"[DATA_VALIDATION_ERROR] {expected_msg}"
        assert exc.field_name == "age"
        assert isinstance(exc, DataValidationError)

    @staticmethod
    def test_create_validation_error_different_types() -> None:
        """Test create_validation_error with different value types."""
        # Test with list value
        exc = create_validation_error("scores", "dict", [1, 2, 3])
        assert "expected dict, got list" in str(exc)

        # Test with None value
        exc = create_validation_error("data", "str", None)
        assert "expected str, got NoneType" in str(exc)

    @staticmethod
    def test_create_numeric_validation_error() -> None:
        """Test create_numeric_validation_error utility function."""
        exc = create_numeric_validation_error("heart_rate", "invalid")

        assert exc.data_type == "heart_rate"
        assert exc.expected_type == "numeric (int or float)"
        assert exc.actual_type == "str"
        assert isinstance(exc, InvalidNHANESStatsDataError)

    @staticmethod
    def test_create_numeric_validation_error_different_types() -> None:
        """Test create_numeric_validation_error with different value types."""
        # Test with list
        exc = create_numeric_validation_error("blood_pressure", [120, 80])
        assert exc.actual_type == "list"

        # Test with dict
        exc = create_numeric_validation_error("temperature", {"value": 98.6})
        assert exc.actual_type == "dict"


class TestExceptionInheritance:
    """Tests for exception inheritance hierarchy."""

    @staticmethod
    def test_exception_inheritance_hierarchy() -> None:
        """Test that all exceptions inherit from correct base classes."""
        # Test that all custom exceptions inherit from ClarityBaseError
        assert issubclass(DataValidationError, ClarityBaseError)
        assert issubclass(ModelError, ClarityBaseError)
        assert issubclass(ServiceError, ClarityBaseError)
        assert issubclass(AuthenticationError, ClarityBaseError)
        assert issubclass(ConfigurationError, ClarityBaseError)

        # Test specific inheritance chains
        assert issubclass(InvalidStepCountDataError, DataValidationError)
        assert issubclass(ModelNotInitializedError, ModelError)
        assert issubclass(InferenceError, ModelError)
        assert issubclass(InferenceTimeoutError, InferenceError)
        assert issubclass(AccountDisabledError, AuthenticationError)
        assert issubclass(AccessDeniedError, AuthorizationError)

    @staticmethod
    def test_exception_isinstance_checks() -> None:
        """Test isinstance checks work correctly."""
        # Create specific exceptions
        validation_exc = InvalidStepCountDataError("Invalid step count")
        inference_exc = InferenceTimeoutError("req123", 30.0)
        auth_exc = AccountDisabledError("user123")

        # Test isinstance with base classes
        assert isinstance(validation_exc, DataValidationError)
        assert isinstance(validation_exc, ClarityBaseError)
        assert isinstance(inference_exc, InferenceError)
        assert isinstance(inference_exc, ModelError)
        assert isinstance(auth_exc, AuthenticationError)
        assert isinstance(auth_exc, ClarityBaseError)

    @staticmethod
    def test_exception_error_codes() -> None:
        """Test that exceptions have correct error codes."""
        # Test various exceptions have expected error codes
        assert DataValidationError("test").error_code == "DATA_VALIDATION_ERROR"
        assert IntegrationError("test").error_code == "INTEGRATION_ERROR"
        assert ModelNotInitializedError().error_code == "MODEL_NOT_INITIALIZED"
        assert InferenceError("test").error_code == "INFERENCE_ERROR"
        assert NHANESStatsError("test").error_code == "NHANES_STATS_ERROR"
        assert (
            ServiceNotInitializedError("test").error_code == "SERVICE_NOT_INITIALIZED"
        )
        assert AuthenticationError("test").error_code == "AUTHENTICATION_ERROR"
        assert ConfigurationError("test").error_code == "CONFIGURATION_ERROR"
        assert CacheError("test").error_code == "CACHE_ERROR"


class TestExceptionEdgeCases:
    """Tests for edge cases and error conditions."""

    @staticmethod
    def test_exception_with_complex_details() -> None:
        """Test exceptions with complex details objects."""
        complex_details = {
            "validation_errors": [
                {"field": "email", "code": "INVALID_FORMAT"},
                {"field": "age", "code": "OUT_OF_RANGE"},
            ],
            "request_context": {
                "user_id": "user123",
                "endpoint": "/api/v1/users",
                "timestamp": "2024-01-01T00:00:00Z",
            },
            "debug_info": {
                "stack_trace": ["frame1", "frame2"],
                "memory_usage": 1024,
            },
        }

        exc = DataValidationError("Complex validation failed", details=complex_details)

        assert exc.details == complex_details
        assert exc.details["validation_errors"][0]["field"] == "email"
        assert exc.details["request_context"]["user_id"] == "user123"

    @staticmethod
    def test_exception_string_representation() -> None:
        """Test string representation of exceptions."""
        # Test exception without error code
        exc1 = ClarityBaseError("Simple error")
        assert str(exc1) == "Simple error"

        # Test exception with error code
        exc2 = DataValidationError("Validation failed")
        assert str(exc2) == "[DATA_VALIDATION_ERROR] Validation failed"

        # Test exception with very long message
        long_message = "A" * 1000
        exc3 = ClarityBaseError(long_message, error_code="LONG_ERROR")
        assert str(exc3) == f"[LONG_ERROR] {long_message}"

    @staticmethod
    def test_exception_with_special_characters() -> None:
        """Test exceptions with special characters in messages."""
        special_msg = "Error with émojis 🚀 and unicode characters: ñáéíóú"
        exc = DataValidationError(special_msg)

        assert special_msg in str(exc)
        assert exc.error_code == "DATA_VALIDATION_ERROR"

    @staticmethod
    def test_api_exception_with_empty_values() -> None:
        """Test ClarityAPIException with empty/None values."""
        exc = ClarityAPIException(
            status_code=400,
            problem_type="",
            title="",
            detail="",
            instance="",
            trace_id="",
            errors=[],
            help_url="",
        )

        problem = exc.to_problem_detail()
        assert not problem.type
        assert not problem.title
        assert not problem.detail
        assert problem.errors == []
