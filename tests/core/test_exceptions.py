"""Comprehensive tests for core exceptions functionality.

Tests cover:
- RFC 7807 Problem Details implementation
- ClarityAPIException hierarchy
- Exception handlers for FastAPI
- Domain-specific business logic exceptions
- Error utility functions
- Edge cases and error scenarios
"""

from __future__ import annotations

import json
from unittest.mock import Mock, patch
from uuid import UUID

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import pytest

from clarity.core.exceptions import (  # Auth exceptions; Cache exceptions; Base exceptions; Configuration exceptions; Data validation exceptions; ML and model exceptions; NHANES exceptions; Service exceptions; API Problem types; Utility functions; Exception handlers
    AccessDeniedError,
    AccountDisabledError,
    AuthenticationError,
    AuthenticationProblem,
    AuthorizationError,
    AuthorizationProblem,
    CacheError,
    CacheKeyError,
    ClarityAPIException,
    ClarityBaseError,
    ConfigurationError,
    ConflictProblem,
    DataLengthMismatchError,
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
    ModelError,
    ModelNotInitializedError,
    NHANESDataNotFoundError,
    NHANESStatsError,
    ProblemDetail,
    ProcessingError,
    RateLimitProblem,
    ResourceNotFoundProblem,
    ServiceError,
    ServiceNotInitializedError,
    ServiceUnavailableError,
    ServiceUnavailableProblem,
    ValidationProblem,
    create_numeric_validation_error,
    create_validation_error,
    generic_exception_handler,
    problem_detail_exception_handler,
)


class TestProblemDetail:
    """Test RFC 7807 Problem Detail model."""

    @staticmethod
    def test_problem_detail_creation() -> None:
        """Test creating Problem Detail with all fields."""
        problem = ProblemDetail(
            type="https://api.clarity.health/problems/validation-error",
            title="Validation Error",
            status=400,
            detail="The submitted health data contains invalid heart rate values",
            instance="https://api.clarity.health/requests/550e8400-e29b-41d4-a716-446655440000",
            trace_id="550e8400-e29b-41d4-a716-446655440000",
            errors=[
                {
                    "field": "heart_rate",
                    "message": "Invalid value",
                    "code": "INVALID_RANGE",
                }
            ],
            help_url="https://docs.clarity.health/errors/validation-error",
        )

        assert problem.type == "https://api.clarity.health/problems/validation-error"
        assert problem.title == "Validation Error"
        assert problem.status == 400
        assert (
            problem.detail
            == "The submitted health data contains invalid heart rate values"
        )
        assert (
            problem.instance
            == "https://api.clarity.health/requests/550e8400-e29b-41d4-a716-446655440000"
        )
        assert problem.trace_id == "550e8400-e29b-41d4-a716-446655440000"
        assert problem.errors == [
            {"field": "heart_rate", "message": "Invalid value", "code": "INVALID_RANGE"}
        ]
        assert problem.help_url == "https://docs.clarity.health/errors/validation-error"

    @staticmethod
    def test_problem_detail_minimal() -> None:
        """Test creating Problem Detail with minimal required fields."""
        problem = ProblemDetail(
            type="https://api.clarity.health/problems/generic",
            title="Generic Error",
            status=500,
            detail="An error occurred",
            instance="https://api.clarity.health/requests/test-123",
        )

        assert problem.type == "https://api.clarity.health/problems/generic"
        assert problem.title == "Generic Error"
        assert problem.status == 500
        assert problem.detail == "An error occurred"
        assert problem.instance == "https://api.clarity.health/requests/test-123"
        assert problem.trace_id is None
        assert problem.errors is None
        assert problem.help_url is None

    @staticmethod
    def test_problem_detail_serialization() -> None:
        """Test Problem Detail serialization."""
        problem = ProblemDetail(
            type="https://api.clarity.health/problems/test",
            title="Test Error",
            status=400,
            detail="Test detail",
            instance="https://api.clarity.health/requests/test",
        )

        serialized = problem.model_dump(exclude_none=True)
        assert "type" in serialized
        assert "title" in serialized
        assert "status" in serialized
        assert "detail" in serialized
        assert "instance" in serialized
        assert "trace_id" not in serialized  # Should be excluded since it's None


class TestClarityAPIException:
    """Test ClarityAPIException base class."""

    @staticmethod
    def test_clarity_api_exception_creation() -> None:
        """Test creating ClarityAPIException with all parameters."""
        exception = ClarityAPIException(
            status_code=400,
            problem_type="https://api.clarity.health/problems/test",
            title="Test Error",
            detail="Test detail message",
            instance="https://api.clarity.health/requests/test-123",
            trace_id="trace-123",
            errors=[{"field": "test", "message": "Test error"}],
            help_url="https://docs.clarity.health/errors/test",
            headers={"X-Custom": "test"},
        )

        assert exception.status_code == 400
        assert exception.problem_type == "https://api.clarity.health/problems/test"
        assert exception.title == "Test Error"
        assert exception.detail == "Test detail message"
        assert exception.instance == "https://api.clarity.health/requests/test-123"
        assert exception.trace_id == "trace-123"
        assert exception.errors == [{"field": "test", "message": "Test error"}]
        assert exception.help_url == "https://docs.clarity.health/errors/test"
        assert exception.headers == {"X-Custom": "test"}

    @staticmethod
    def test_clarity_api_exception_auto_generation() -> None:
        """Test ClarityAPIException with auto-generated fields."""
        exception = ClarityAPIException(
            status_code=500,
            problem_type="https://api.clarity.health/problems/internal",
            title="Internal Error",
            detail="Internal server error occurred",
        )

        assert exception.status_code == 500
        assert exception.problem_type == "https://api.clarity.health/problems/internal"
        assert exception.title == "Internal Error"
        assert exception.detail == "Internal server error occurred"

        # Auto-generated fields
        assert exception.instance.startswith("https://api.clarity.health/requests/")
        assert len(exception.trace_id) > 0

        # UUID validation
        instance_id = exception.instance.split("/")[-1]
        trace_id = exception.trace_id
        assert UUID(instance_id)  # Should not raise
        assert UUID(trace_id)  # Should not raise

    @staticmethod
    def test_to_problem_detail() -> None:
        """Test converting ClarityAPIException to Problem Detail."""
        exception = ClarityAPIException(
            status_code=422,
            problem_type="https://api.clarity.health/problems/validation",
            title="Validation Failed",
            detail="Request validation failed",
            trace_id="test-trace-id",
        )

        problem = exception.to_problem_detail()

        assert isinstance(problem, ProblemDetail)
        assert problem.type == "https://api.clarity.health/problems/validation"
        assert problem.title == "Validation Failed"
        assert problem.status == 422
        assert problem.detail == "Request validation failed"
        assert problem.trace_id == "test-trace-id"

    @staticmethod
    def test_clarity_api_exception_inheritance() -> None:
        """Test that ClarityAPIException properly inherits from HTTPException."""
        exception = ClarityAPIException(
            status_code=404,
            problem_type="https://api.clarity.health/problems/not-found",
            title="Not Found",
            detail="Resource not found",
        )

        assert isinstance(exception, HTTPException)
        assert exception.status_code == 404
        assert exception.detail == "Resource not found"


class TestPredefinedProblemTypes:
    """Test predefined Problem Detail exception types."""

    @staticmethod
    def test_validation_problem() -> None:
        """Test ValidationProblem creation and properties."""
        errors = [{"field": "email", "message": "Invalid email format"}]
        exception = ValidationProblem(
            detail="Validation failed for request", errors=errors, trace_id="test-trace"
        )

        assert exception.status_code == 400
        assert (
            exception.problem_type
            == "https://api.clarity.health/problems/validation-error"
        )
        assert exception.title == "Validation Error"
        assert exception.detail == "Validation failed for request"
        assert exception.errors == errors
        assert exception.trace_id == "test-trace"
        assert exception.help_url == "https://docs.clarity.health/errors/validation"

    @staticmethod
    def test_authentication_problem() -> None:
        """Test AuthenticationProblem creation and properties."""
        exception = AuthenticationProblem(
            detail="Invalid credentials provided", trace_id="auth-trace"
        )

        assert exception.status_code == 401
        assert (
            exception.problem_type
            == "https://api.clarity.health/problems/authentication-required"
        )
        assert exception.title == "Authentication Required"
        assert exception.detail == "Invalid credentials provided"
        assert exception.trace_id == "auth-trace"
        assert exception.help_url == "https://docs.clarity.health/authentication"

    @staticmethod
    def test_authentication_problem_default() -> None:
        """Test AuthenticationProblem with default message."""
        exception = AuthenticationProblem()

        assert exception.detail == "Authentication required"

    @staticmethod
    def test_authorization_problem() -> None:
        """Test AuthorizationProblem creation and properties."""
        exception = AuthorizationProblem(
            detail="Access denied to this resource", trace_id="authz-trace"
        )

        assert exception.status_code == 403
        assert (
            exception.problem_type
            == "https://api.clarity.health/problems/authorization-denied"
        )
        assert exception.title == "Authorization Denied"
        assert exception.detail == "Access denied to this resource"
        assert exception.help_url == "https://docs.clarity.health/permissions"

    @staticmethod
    def test_authorization_problem_default() -> None:
        """Test AuthorizationProblem with default message."""
        exception = AuthorizationProblem()

        assert exception.detail == "Insufficient permissions for this resource"

    @staticmethod
    def test_resource_not_found_problem() -> None:
        """Test ResourceNotFoundProblem creation and properties."""
        exception = ResourceNotFoundProblem(
            resource_type="User", resource_id="user-123", trace_id="not-found-trace"
        )

        assert exception.status_code == 404
        assert (
            exception.problem_type
            == "https://api.clarity.health/problems/resource-not-found"
        )
        assert exception.title == "Resource Not Found"
        assert exception.detail == "User with ID 'user-123' does not exist"
        assert exception.trace_id == "not-found-trace"
        assert exception.help_url == "https://docs.clarity.health/errors/not-found"

    @staticmethod
    def test_conflict_problem() -> None:
        """Test ConflictProblem creation and properties."""
        exception = ConflictProblem(
            detail="Resource already exists with this identifier",
            trace_id="conflict-trace",
        )

        assert exception.status_code == 409
        assert (
            exception.problem_type
            == "https://api.clarity.health/problems/resource-conflict"
        )
        assert exception.title == "Resource Conflict"
        assert exception.detail == "Resource already exists with this identifier"
        assert exception.help_url == "https://docs.clarity.health/errors/conflict"

    @staticmethod
    def test_rate_limit_problem() -> None:
        """Test RateLimitProblem creation and properties."""
        exception = RateLimitProblem(
            retry_after=60, detail="Too many requests", trace_id="rate-limit-trace"
        )

        assert exception.status_code == 429
        assert (
            exception.problem_type
            == "https://api.clarity.health/problems/rate-limit-exceeded"
        )
        assert exception.title == "Rate Limit Exceeded"
        assert exception.detail == "Too many requests"
        assert exception.headers == {"Retry-After": "60"}
        assert exception.help_url == "https://docs.clarity.health/rate-limits"

    @staticmethod
    def test_rate_limit_problem_default() -> None:
        """Test RateLimitProblem with default message."""
        exception = RateLimitProblem(retry_after=30)

        assert exception.detail == "Rate limit exceeded"

    @staticmethod
    def test_internal_server_problem() -> None:
        """Test InternalServerProblem creation and properties."""
        exception = InternalServerProblem(
            detail="Database connection failed", trace_id="internal-trace"
        )

        assert exception.status_code == 500
        assert (
            exception.problem_type
            == "https://api.clarity.health/problems/internal-server-error"
        )
        assert exception.title == "Internal Server Error"
        assert exception.detail == "Database connection failed"
        assert exception.help_url == "https://docs.clarity.health/errors/server-error"

    @staticmethod
    def test_internal_server_problem_default() -> None:
        """Test InternalServerProblem with default message."""
        exception = InternalServerProblem()

        assert exception.detail == "An internal server error occurred"

    @staticmethod
    def test_service_unavailable_problem() -> None:
        """Test ServiceUnavailableProblem creation and properties."""
        exception = ServiceUnavailableProblem(
            service_name="Database Service", retry_after=120, trace_id="service-trace"
        )

        assert exception.status_code == 503
        assert (
            exception.problem_type
            == "https://api.clarity.health/problems/service-unavailable"
        )
        assert exception.title == "Service Unavailable"
        assert exception.detail == "Database Service is temporarily unavailable"
        assert exception.headers == {"Retry-After": "120"}
        assert (
            exception.help_url
            == "https://docs.clarity.health/errors/service-unavailable"
        )

    @staticmethod
    def test_service_unavailable_problem_no_retry() -> None:
        """Test ServiceUnavailableProblem without retry-after."""
        exception = ServiceUnavailableProblem(service_name="ML Service")

        assert exception.detail == "ML Service is temporarily unavailable"
        assert exception.headers is None


class TestExceptionHandlers:
    """Test FastAPI exception handlers."""

    @staticmethod
    def test_problem_detail_exception_handler() -> None:
        """Test problem detail exception handler."""
        request = Mock(spec=Request)
        exception = ValidationProblem(
            detail="Test validation error",
            errors=[{"field": "test", "message": "Test error"}],
        )

        response = problem_detail_exception_handler(request, exception)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400

        # Parse response content
        content = json.loads(response.body)
        assert content["type"] == "https://api.clarity.health/problems/validation-error"
        assert content["title"] == "Validation Error"
        assert content["status"] == 400
        assert content["detail"] == "Test validation error"
        assert content["errors"] == [{"field": "test", "message": "Test error"}]

    @staticmethod
    def test_problem_detail_exception_handler_with_headers() -> None:
        """Test problem detail exception handler with custom headers."""
        request = Mock(spec=Request)
        exception = RateLimitProblem(retry_after=60, detail="Rate limit exceeded")

        response = problem_detail_exception_handler(request, exception)

        assert response.status_code == 429
        assert response.headers["Retry-After"] == "60"

    @staticmethod
    @patch("clarity.core.exceptions.logger")
    def test_generic_exception_handler(mock_logger: Mock) -> None:
        """Test generic exception handler for unexpected exceptions."""
        request = Mock(spec=Request)
        exception = ValueError("Unexpected error")

        response = generic_exception_handler(request, exception)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 500

        # Parse response content
        content = json.loads(response.body)
        assert (
            content["type"]
            == "https://api.clarity.health/problems/internal-server-error"
        )
        assert content["title"] == "Internal Server Error"
        assert content["status"] == 500
        assert content["detail"] == "An unexpected error occurred"
        assert "trace_id" in content

        # Verify logging
        mock_logger.error.assert_called_once()
        log_call_args = mock_logger.error.call_args[0]
        assert "Unhandled exception" in log_call_args[0]
        assert exception == mock_logger.error.call_args[1]["exc_info"]


class TestClarityBaseError:
    """Test ClarityBaseError base exception class."""

    @staticmethod
    def test_clarity_base_error_basic() -> None:
        """Test basic ClarityBaseError creation."""
        error = ClarityBaseError("Test error message")

        assert str(error) == "Test error message"
        assert error.error_code is None
        assert error.details == {}

    @staticmethod
    def test_clarity_base_error_with_code() -> None:
        """Test ClarityBaseError with error code."""
        error = ClarityBaseError(
            "Test error message", error_code="TEST_ERROR", details={"key": "value"}
        )

        assert str(error) == "[TEST_ERROR] Test error message"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {"key": "value"}

    @staticmethod
    def test_clarity_base_error_inheritance() -> None:
        """Test that ClarityBaseError inherits from Exception."""
        error = ClarityBaseError("Test message")
        assert isinstance(error, Exception)


class TestDataValidationExceptions:
    """Test data validation exception classes."""

    @staticmethod
    def test_data_validation_error() -> None:
        """Test DataValidationError creation and properties."""
        error = DataValidationError(
            "Invalid data format",
            field_name="heart_rate",
            details={"min": 0, "max": 300},
        )

        assert str(error) == "[DATA_VALIDATION_ERROR] Invalid data format"
        assert error.error_code == "DATA_VALIDATION_ERROR"
        assert error.field_name == "heart_rate"
        assert error.details == {"min": 0, "max": 300}

    @staticmethod
    def test_invalid_step_count_data_error() -> None:
        """Test InvalidStepCountDataError inherits from DataValidationError."""
        error = InvalidStepCountDataError("Invalid step count")

        assert isinstance(error, DataValidationError)
        assert error.error_code == "DATA_VALIDATION_ERROR"

    @staticmethod
    def test_invalid_nhanes_stats_error() -> None:
        """Test InvalidNHANESStatsError inherits from DataValidationError."""
        error = InvalidNHANESStatsError("Invalid NHANES stats")

        assert isinstance(error, DataValidationError)
        assert error.error_code == "DATA_VALIDATION_ERROR"

    @staticmethod
    def test_processing_error() -> None:
        """Test ProcessingError inherits from DataValidationError."""
        error = ProcessingError("Processing failed")

        assert isinstance(error, DataValidationError)
        assert error.error_code == "DATA_VALIDATION_ERROR"

    @staticmethod
    def test_integration_error() -> None:
        """Test IntegrationError creation."""
        error = IntegrationError(
            "API integration failed",
            details={"service": "external_api", "status_code": 500},
        )

        assert str(error) == "[INTEGRATION_ERROR] API integration failed"
        assert error.error_code == "INTEGRATION_ERROR"
        assert error.details == {"service": "external_api", "status_code": 500}

    @staticmethod
    def test_data_length_mismatch_error() -> None:
        """Test DataLengthMismatchError creation and properties."""
        error = DataLengthMismatchError(
            expected_length=100, actual_length=95, data_type="heart_rate_samples"
        )

        assert "length mismatch" in str(error)
        assert "expected 100" in str(error)
        assert "got 95" in str(error)
        assert "heart_rate_samples" in str(error)
        assert error.expected_length == 100
        assert error.actual_length == 95
        assert error.data_type == "heart_rate_samples"

    @staticmethod
    def test_empty_data_error() -> None:
        """Test EmptyDataError creation and properties."""
        error = EmptyDataError("sensor_readings")

        assert "sensor_readings cannot be empty" in str(error)
        assert error.data_type == "sensor_readings"

    @staticmethod
    def test_empty_data_error_default() -> None:
        """Test EmptyDataError with default data type."""
        error = EmptyDataError()

        assert "data cannot be empty" in str(error)
        assert error.data_type == "data"


class TestMLModelExceptions:
    """Test ML model and inference exception classes."""

    @staticmethod
    def test_model_error() -> None:
        """Test ModelError base class."""
        error = ModelError("Model operation failed")

        assert isinstance(error, ClarityBaseError)
        assert str(error) == "Model operation failed"

    @staticmethod
    def test_model_not_initialized_error() -> None:
        """Test ModelNotInitializedError creation and properties."""
        error = ModelNotInitializedError("PAT Model")

        assert isinstance(error, ModelError)
        assert "PAT Model is not initialized" in str(error)
        assert error.error_code == "MODEL_NOT_INITIALIZED"
        assert error.model_name == "PAT Model"

    @staticmethod
    def test_model_not_initialized_error_default() -> None:
        """Test ModelNotInitializedError with default model name."""
        error = ModelNotInitializedError()

        assert "Model is not initialized" in str(error)
        assert error.model_name == "Model"

    @staticmethod
    def test_inference_error() -> None:
        """Test InferenceError creation and properties."""
        error = InferenceError(
            "Inference computation failed",
            request_id="req-123",
            details={"model": "PAT", "version": "1.0"},
        )

        assert isinstance(error, ModelError)
        assert error.error_code == "INFERENCE_ERROR"
        assert error.request_id == "req-123"
        assert error.details == {"model": "PAT", "version": "1.0"}

    @staticmethod
    def test_inference_timeout_error() -> None:
        """Test InferenceTimeoutError creation and properties."""
        error = InferenceTimeoutError(request_id="req-456", timeout_seconds=30.0)

        assert isinstance(error, InferenceError)
        assert "req-456" in str(error)
        assert "timed out after 30.0s" in str(error)
        assert error.request_id == "req-456"
        assert error.timeout_seconds == 30.0


class TestNHANESExceptions:
    """Test NHANES statistics exception classes."""

    @staticmethod
    def test_nhanes_stats_error() -> None:
        """Test NHANESStatsError creation."""
        error = NHANESStatsError(
            "NHANES lookup failed", details={"year": 2020, "age_group": "adult"}
        )

        assert isinstance(error, ClarityBaseError)
        assert error.error_code == "NHANES_STATS_ERROR"
        assert error.details == {"year": 2020, "age_group": "adult"}

    @staticmethod
    def test_nhanes_data_not_found_error() -> None:
        """Test NHANESDataNotFoundError with all parameters."""
        error = NHANESDataNotFoundError(year=2020, age_group="adult", sex="male")

        assert isinstance(error, NHANESStatsError)
        assert "year=2020" in str(error)
        assert "age_group=adult" in str(error)
        assert "sex=male" in str(error)
        assert error.year == 2020
        assert error.age_group == "adult"
        assert error.sex == "male"

    @staticmethod
    def test_nhanes_data_not_found_error_partial() -> None:
        """Test NHANESDataNotFoundError with partial parameters."""
        error = NHANESDataNotFoundError(year=2018)

        assert "year=2018" in str(error)
        assert error.year == 2018
        assert error.age_group is None
        assert error.sex is None

    @staticmethod
    def test_nhanes_data_not_found_error_empty() -> None:
        """Test NHANESDataNotFoundError with no parameters."""
        error = NHANESDataNotFoundError()

        assert "specified criteria" in str(error)
        assert error.year is None
        assert error.age_group is None
        assert error.sex is None

    @staticmethod
    def test_invalid_nhanes_stats_data_error() -> None:
        """Test InvalidNHANESStatsDataError creation and properties."""
        error = InvalidNHANESStatsDataError(
            data_type="step_count", expected_type="numeric", actual_type="string"
        )

        assert isinstance(error, NHANESStatsError)
        assert "Invalid NHANES step_count" in str(error)
        assert "expected numeric" in str(error)
        assert "got string" in str(error)
        assert error.data_type == "step_count"
        assert error.expected_type == "numeric"
        assert error.actual_type == "string"


class TestServiceExceptions:
    """Test service-level exception classes."""

    @staticmethod
    def test_service_error() -> None:
        """Test ServiceError base class."""
        error = ServiceError("Service operation failed")

        assert isinstance(error, ClarityBaseError)

    @staticmethod
    def test_service_not_initialized_error() -> None:
        """Test ServiceNotInitializedError creation and properties."""
        error = ServiceNotInitializedError("Authentication Service")

        assert isinstance(error, ServiceError)
        assert "Authentication Service service is not initialized" in str(error)
        assert error.error_code == "SERVICE_NOT_INITIALIZED"
        assert error.service_name == "Authentication Service"

    @staticmethod
    def test_service_unavailable_error() -> None:
        """Test ServiceUnavailableError creation and properties."""
        error = ServiceUnavailableError(
            service_name="Database Service", reason="Connection timeout"
        )

        assert isinstance(error, ServiceError)
        assert "Database Service service is unavailable" in str(error)
        assert "Connection timeout" in str(error)
        assert error.error_code == "SERVICE_UNAVAILABLE"
        assert error.service_name == "Database Service"
        assert error.reason == "Connection timeout"

    @staticmethod
    def test_service_unavailable_error_no_reason() -> None:
        """Test ServiceUnavailableError without reason."""
        error = ServiceUnavailableError("ML Service")

        assert "ML Service service is unavailable" in str(error)
        assert error.reason is None


class TestAuthExceptions:
    """Test authentication and authorization exception classes."""

    @staticmethod
    def test_authentication_error() -> None:
        """Test AuthenticationError base class."""
        error = AuthenticationError(
            "Token validation failed",
            details={"token_type": "JWT", "issuer": "clarity"},
        )

        assert isinstance(error, ClarityBaseError)
        assert error.error_code == "AUTHENTICATION_ERROR"
        assert error.details == {"token_type": "JWT", "issuer": "clarity"}

    @staticmethod
    def test_authorization_error() -> None:
        """Test AuthorizationError base class."""
        error = AuthorizationError(
            "Insufficient permissions",
            details={"required_role": "admin", "user_role": "user"},
        )

        assert isinstance(error, ClarityBaseError)
        assert error.error_code == "AUTHORIZATION_ERROR"
        assert error.details == {"required_role": "admin", "user_role": "user"}

    @staticmethod
    def test_account_disabled_error() -> None:
        """Test AccountDisabledError creation and properties."""
        error = AccountDisabledError("user-123")

        assert isinstance(error, AuthenticationError)
        assert "User account user-123 is disabled" in str(error)
        assert error.user_id == "user-123"

    @staticmethod
    def test_access_denied_error() -> None:
        """Test AccessDeniedError creation and properties."""
        error = AccessDeniedError(resource="health_data", user_id="user-456")

        assert isinstance(error, AuthorizationError)
        assert "Access denied to health_data" in str(error)
        assert "for user user-456" in str(error)
        assert error.resource == "health_data"
        assert error.user_id == "user-456"

    @staticmethod
    def test_access_denied_error_no_user() -> None:
        """Test AccessDeniedError without user ID."""
        error = AccessDeniedError("admin_panel")

        assert "Access denied to admin_panel" in str(error)
        assert error.resource == "admin_panel"
        assert error.user_id is None


class TestConfigurationExceptions:
    """Test configuration exception classes."""

    @staticmethod
    def test_configuration_error() -> None:
        """Test ConfigurationError creation and properties."""
        error = ConfigurationError(
            "Database connection string invalid",
            config_key="DATABASE_URL",
            details={"section": "database"},
        )

        assert isinstance(error, ClarityBaseError)
        assert error.error_code == "CONFIGURATION_ERROR"
        assert error.config_key == "DATABASE_URL"
        assert error.details == {"section": "database"}

    @staticmethod
    def test_missing_configuration_error() -> None:
        """Test MissingConfigurationError creation and properties."""
        error = MissingConfigurationError("API_KEY")

        assert isinstance(error, ConfigurationError)
        assert "Missing required configuration: API_KEY" in str(error)
        assert error.config_key == "API_KEY"

    @staticmethod
    def test_invalid_configuration_error() -> None:
        """Test InvalidConfigurationError creation and properties."""
        error = InvalidConfigurationError(
            config_key="MAX_CONNECTIONS", value=-5, reason="must be positive"
        )

        assert isinstance(error, ConfigurationError)
        assert "Invalid configuration value for MAX_CONNECTIONS" in str(error)
        assert "-5" in str(error)
        assert "must be positive" in str(error)
        assert error.config_key == "MAX_CONNECTIONS"
        assert error.value == -5
        assert error.reason == "must be positive"

    @staticmethod
    def test_invalid_configuration_error_no_reason() -> None:
        """Test InvalidConfigurationError without reason."""
        error = InvalidConfigurationError(config_key="TIMEOUT", value="invalid")

        assert "Invalid configuration value for TIMEOUT" in str(error)
        assert "invalid" in str(error)
        assert error.reason is None


class TestCacheExceptions:
    """Test cache-related exception classes."""

    @staticmethod
    def test_cache_error() -> None:
        """Test CacheError creation."""
        error = CacheError(
            "Redis connection failed", details={"host": "localhost", "port": 6379}
        )

        assert isinstance(error, ClarityBaseError)
        assert error.error_code == "CACHE_ERROR"
        assert error.details == {"host": "localhost", "port": 6379}

    @staticmethod
    def test_cache_key_error() -> None:
        """Test CacheKeyError creation and properties."""
        error = CacheKeyError(cache_key="user:123:profile", operation="get")

        assert isinstance(error, CacheError)
        assert "Cache get failed for key: user:123:profile" in str(error)
        assert error.cache_key == "user:123:profile"
        assert error.operation == "get"


class TestUtilityFunctions:
    """Test exception utility functions."""

    @staticmethod
    def test_create_validation_error() -> None:
        """Test create_validation_error utility function."""
        error = create_validation_error(
            field_name="age", expected_type="integer", actual_value="twenty"
        )

        assert isinstance(error, DataValidationError)
        assert "Field 'age' expected integer" in str(error)
        assert "got str: twenty" in str(error)
        assert error.field_name == "age"

    @staticmethod
    def test_create_numeric_validation_error() -> None:
        """Test create_numeric_validation_error utility function."""
        error = create_numeric_validation_error(
            field_name="heart_rate", value="not_a_number"
        )

        assert isinstance(error, InvalidNHANESStatsDataError)
        assert error.data_type == "heart_rate"
        assert error.expected_type == "numeric (int or float)"
        assert error.actual_type == "str"


class TestExceptionEdgeCases:
    """Test edge cases and integration scenarios for exceptions."""

    @staticmethod
    def test_exception_chaining() -> None:
        """Test exception chaining preserves context."""

        def raise_value_error() -> None:
            msg = "Original error"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="Original error") as exc_info:
            raise_value_error()

        original = exc_info.value
        new_error = InferenceError(
            "Inference failed due to validation", request_id="req-789"
        )

        # Test that we can chain exceptions
        assert original is not None
        assert isinstance(new_error, InferenceError)

    @staticmethod
    def test_exception_with_complex_details() -> None:
        """Test exceptions with complex detail objects."""
        complex_details = {
            "request": {
                "url": "/api/v1/health-data",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
            },
            "validation_errors": [
                {"field": "heart_rate", "value": -10, "constraint": "positive"},
                {"field": "timestamp", "value": "invalid", "constraint": "iso8601"},
            ],
            "metadata": {
                "user_id": "user-123",
                "request_id": "req-456",
                "timestamp": "2025-01-15T10:30:00Z",
            },
        }

        error = ValidationProblem(
            detail="Complex validation failure",
            errors=complex_details["validation_errors"],  # type: ignore[arg-type]
        )

        problem = error.to_problem_detail()
        assert problem.errors == complex_details["validation_errors"]

    @staticmethod
    def test_problem_detail_exclude_none_serialization() -> None:
        """Test Problem Detail serialization excludes None values."""
        problem = ProblemDetail(
            type="https://api.clarity.health/problems/test",
            title="Test",
            status=400,
            detail="Test detail",
            instance="https://api.clarity.health/requests/test",
            trace_id=None,  # Should be excluded
            errors=None,  # Should be excluded
            help_url=None,  # Should be excluded
        )

        serialized = problem.model_dump(exclude_none=True)
        assert "trace_id" not in serialized
        assert "errors" not in serialized
        assert "help_url" not in serialized
        assert len(serialized) == 5  # Only non-None fields

    @staticmethod
    def test_exception_handler_with_no_headers() -> None:
        """Test exception handler with exception that has no headers."""
        request = Mock(spec=Request)
        exception = ValidationProblem("Test error")

        response = problem_detail_exception_handler(request, exception)

        # Should not include headers in response if exception has none
        assert response.headers.get("Retry-After") is None
