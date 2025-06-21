"""Custom exception hierarchy for the Clarity Digital Twin platform.

This module provides a comprehensive exception hierarchy following clean code principles,
enabling precise error handling and meaningful error messages throughout the application.

The exception hierarchy is designed to be:
- Specific and descriptive
- Easy to catch and handle at appropriate levels
- Consistent in structure and naming
- Self-documenting through clear names and messages
"""

# removed - breaks FastAPI

import logging
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Module-level logger for exception handling
logger = logging.getLogger(__name__)


class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details for HTTP APIs.

    Professional-grade error responses with structured debugging info.
    """

    type: str = Field(
        ...,
        description="URI reference identifying the problem type",
        examples=["https://api.clarity.health/problems/validation-error"],
    )
    title: str = Field(
        ...,
        description="Short, human-readable summary of the problem",
        examples=["Validation Error"],
    )
    status: int = Field(..., description="HTTP status code", examples=[400])
    detail: str = Field(
        ...,
        description="Human-readable explanation specific to this occurrence",
        examples=["The submitted health data contains invalid heart rate values"],
    )
    instance: str = Field(
        ...,
        description="URI reference identifying this specific occurrence",
        examples=[
            "https://api.clarity.health/requests/550e8400-e29b-41d4-a716-446655440000"
        ],
    )
    trace_id: str | None = Field(
        None,
        description="Distributed tracing identifier for debugging",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    errors: list[dict[str, Any]] | None = Field(
        None, description="Detailed field-level validation errors"
    )
    help_url: str | None = Field(
        None,
        description="URL to documentation for this error type",
        examples=["https://docs.clarity.health/errors/validation-error"],
    )


class ClarityAPIException(HTTPException):
    """🚀 CLARITY Platform Custom Exception with RFC 7807 Support.

    Enterprise-grade exception handling that outputs professional Problem Details.
    """

    def __init__(
        self,
        status_code: int,
        problem_type: str,
        title: str,
        detail: str,
        *,
        instance: str | None = None,
        trace_id: str | None = None,
        errors: list[dict[str, Any]] | None = None,
        help_url: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.problem_type = problem_type
        self.title = title
        self.detail = detail
        self.instance = instance or f"https://api.clarity.health/requests/{uuid4()}"
        self.trace_id = trace_id or str(uuid4())
        self.errors = errors
        self.help_url = help_url

        super().__init__(status_code=status_code, detail=detail, headers=headers)

    def to_problem_detail(self) -> ProblemDetail:
        """Convert to RFC 7807 Problem Detail format."""
        return ProblemDetail(
            type=self.problem_type,
            title=self.title,
            status=self.status_code,
            detail=self.detail,
            instance=self.instance,
            trace_id=self.trace_id,
            errors=self.errors,
            help_url=self.help_url,
        )


# 🔥 Pre-defined Problem Types for Common Scenarios


class ValidationProblem(ClarityAPIException):
    """Validation error with detailed field information."""

    def __init__(
        self,
        detail: str,
        errors: list[dict[str, Any]] | None = None,
        trace_id: str | None = None,
    ) -> None:
        super().__init__(
            status_code=400,
            problem_type="https://api.clarity.health/problems/validation-error",
            title="Validation Error",
            detail=detail,
            errors=errors,
            trace_id=trace_id,
            help_url="https://docs.clarity.health/errors/validation",
        )


class AuthenticationProblem(ClarityAPIException):
    """Authentication required or failed."""

    def __init__(
        self, detail: str = "Authentication required", trace_id: str | None = None
    ) -> None:
        super().__init__(
            status_code=401,
            problem_type="https://api.clarity.health/problems/authentication-required",
            title="Authentication Required",
            detail=detail,
            trace_id=trace_id,
            help_url="https://docs.clarity.health/authentication",
        )


class AuthorizationProblem(ClarityAPIException):
    """Insufficient permissions for requested resource."""

    def __init__(
        self,
        detail: str = "Insufficient permissions for this resource",
        trace_id: str | None = None,
    ) -> None:
        super().__init__(
            status_code=403,
            problem_type="https://api.clarity.health/problems/authorization-denied",
            title="Authorization Denied",
            detail=detail,
            trace_id=trace_id,
            help_url="https://docs.clarity.health/permissions",
        )


class ResourceNotFoundProblem(ClarityAPIException):
    """Requested resource does not exist."""

    def __init__(
        self, resource_type: str, resource_id: str, trace_id: str | None = None
    ) -> None:
        super().__init__(
            status_code=404,
            problem_type="https://api.clarity.health/problems/resource-not-found",
            title="Resource Not Found",
            detail=f"{resource_type} with ID '{resource_id}' does not exist",
            trace_id=trace_id,
            help_url="https://docs.clarity.health/errors/not-found",
        )


class ConflictProblem(ClarityAPIException):
    """Resource conflict (duplicate, state conflict, etc.)."""

    def __init__(self, detail: str, trace_id: str | None = None) -> None:
        super().__init__(
            status_code=409,
            problem_type="https://api.clarity.health/problems/resource-conflict",
            title="Resource Conflict",
            detail=detail,
            trace_id=trace_id,
            help_url="https://docs.clarity.health/errors/conflict",
        )


class RateLimitProblem(ClarityAPIException):
    """Rate limit exceeded."""

    def __init__(
        self,
        retry_after: int,
        detail: str = "Rate limit exceeded",
        trace_id: str | None = None,
    ) -> None:
        headers = {"Retry-After": str(retry_after)}
        super().__init__(
            status_code=429,
            problem_type="https://api.clarity.health/problems/rate-limit-exceeded",
            title="Rate Limit Exceeded",
            detail=detail,
            headers=headers,
            trace_id=trace_id,
            help_url="https://docs.clarity.health/rate-limits",
        )


class InternalServerProblem(ClarityAPIException):
    """Internal server error with trace ID for debugging."""

    def __init__(
        self,
        detail: str = "An internal server error occurred",
        trace_id: str | None = None,
    ) -> None:
        super().__init__(
            status_code=500,
            problem_type="https://api.clarity.health/problems/internal-server-error",
            title="Internal Server Error",
            detail=detail,
            trace_id=trace_id,
            help_url="https://docs.clarity.health/errors/server-error",
        )


class ServiceUnavailableProblem(ClarityAPIException):
    """Service temporarily unavailable."""

    def __init__(
        self,
        service_name: str,
        retry_after: int | None = None,
        trace_id: str | None = None,
        diagnostics: str | None = None,
    ) -> None:
        detail = f"{service_name} is temporarily unavailable"
        if diagnostics:
            detail += f" | Diagnostics: {diagnostics}"
        headers = {"Retry-After": str(retry_after)} if retry_after else None

        super().__init__(
            status_code=503,
            problem_type="https://api.clarity.health/problems/service-unavailable",
            title="Service Unavailable",
            detail=detail,
            headers=headers,
            trace_id=trace_id,
            help_url="https://docs.clarity.health/errors/service-unavailable",
        )


# 🎯 Exception Handler for FastAPI


def problem_detail_exception_handler(
    _request: Request, exc: ClarityAPIException
) -> JSONResponse:
    """Convert ClarityAPIException to RFC 7807 Problem Detail response."""
    from clarity.middleware.security_headers import (  # noqa: PLC0415
        SecurityHeadersMiddleware,
    )

    problem = exc.to_problem_detail()

    response = JSONResponse(
        status_code=exc.status_code,
        content=problem.model_dump(exclude_none=True),
        headers=exc.headers or {},
    )

    # Add security headers to the response
    SecurityHeadersMiddleware.add_security_headers_to_response(response)

    return response


def generic_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with Problem Details format."""
    from clarity.middleware.security_headers import (  # noqa: PLC0415
        SecurityHeadersMiddleware,
    )

    trace_id = str(uuid4())

    problem = InternalServerProblem(
        detail="An unexpected error occurred", trace_id=trace_id
    ).to_problem_detail()

    # Log the actual exception for debugging
    logger.error("Unhandled exception", exc_info=exc)

    response = JSONResponse(
        status_code=500, content=problem.model_dump(exclude_none=True)
    )

    # Add security headers to the response
    SecurityHeadersMiddleware.add_security_headers_to_response(response)

    return response


class ClarityBaseError(Exception):
    """Base exception for all Clarity Digital Twin specific errors.

    This base class provides common functionality for all custom exceptions
    and establishes the foundation for the exception hierarchy.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {super().__str__()}"
        return super().__str__()


# ==============================================================================
# Data Validation Exceptions
# ==============================================================================


class DataValidationError(ClarityBaseError):
    """Raised when data validation fails."""

    def __init__(
        self, message: str, *, field_name: str | None = None, **kwargs: dict[str, Any]
    ) -> None:
        super().__init__(message, error_code="DATA_VALIDATION_ERROR", **kwargs)
        self.field_name = field_name


class InvalidStepCountDataError(DataValidationError):
    """Raised when step count data is invalid or malformed."""


class InvalidNHANESStatsError(DataValidationError):
    """Raised when NHANES statistics data is invalid or malformed."""


class ProcessingError(DataValidationError):
    """Raised when data processing operations fail."""


class IntegrationError(ClarityBaseError):
    """Raised when external integration operations fail."""

    def __init__(self, message: str, **kwargs: dict[str, Any]) -> None:
        super().__init__(message, error_code="INTEGRATION_ERROR", **kwargs)


class DataLengthMismatchError(DataValidationError):
    """Raised when related data arrays have mismatched lengths."""

    def __init__(
        self, expected_length: int, actual_length: int, data_type: str = "data"
    ) -> None:
        message = f"{data_type} length mismatch: expected {expected_length}, got {actual_length}"
        super().__init__(message)
        self.expected_length = expected_length
        self.actual_length = actual_length
        self.data_type = data_type


class EmptyDataError(DataValidationError):
    """Raised when required data is empty."""

    def __init__(self, data_type: str = "data") -> None:
        message = f"{data_type} cannot be empty"
        super().__init__(message)
        self.data_type = data_type


# ==============================================================================
# ML Model and Inference Exceptions
# ==============================================================================


class ModelError(ClarityBaseError):
    """Base class for ML model related errors."""


class ModelNotInitializedError(ModelError):
    """Raised when attempting to use an uninitialized model."""

    def __init__(self, model_name: str = "Model") -> None:
        message = f"{model_name} is not initialized"
        super().__init__(message, error_code="MODEL_NOT_INITIALIZED")
        self.model_name = model_name


class InferenceError(ModelError):
    """Raised when model inference fails."""

    def __init__(
        self, message: str, *, request_id: str | None = None, **kwargs: dict[str, Any]
    ) -> None:
        super().__init__(message, error_code="INFERENCE_ERROR", **kwargs)
        self.request_id = request_id


class InferenceTimeoutError(InferenceError):
    """Raised when inference request times out."""

    def __init__(self, request_id: str, timeout_seconds: float) -> None:
        message = f"Inference request {request_id} timed out after {timeout_seconds}s"
        super().__init__(message, request_id=request_id)
        self.timeout_seconds = timeout_seconds


# ==============================================================================
# NHANES Statistics Exceptions
# ==============================================================================


class NHANESStatsError(ClarityBaseError):
    """Raised when NHANES statistics operations fail."""

    def __init__(self, message: str, **kwargs: dict[str, Any]) -> None:
        super().__init__(message, error_code="NHANES_STATS_ERROR", **kwargs)


class NHANESDataNotFoundError(NHANESStatsError):
    """Raised when requested NHANES data is not available."""

    def __init__(
        self,
        year: int | None = None,
        age_group: str | None = None,
        sex: str | None = None,
    ) -> None:
        parts = []
        if year is not None:
            parts.append(f"year={year}")
        if age_group is not None:
            parts.append(f"age_group={age_group}")
        if sex is not None:
            parts.append(f"sex={sex}")

        criteria = ", ".join(parts) if parts else "specified criteria"
        message = f"NHANES data not found for {criteria}"
        super().__init__(message)
        self.year = year
        self.age_group = age_group
        self.sex = sex


class InvalidNHANESStatsDataError(NHANESStatsError):
    """Raised when NHANES statistics data structure is invalid."""

    def __init__(self, data_type: str, expected_type: str, actual_type: str) -> None:
        message = (
            f"Invalid NHANES {data_type}: expected {expected_type}, got {actual_type}"
        )
        super().__init__(message)
        self.data_type = data_type
        self.expected_type = expected_type
        self.actual_type = actual_type


# ==============================================================================
# Service and Infrastructure Exceptions
# ==============================================================================


class ServiceError(ClarityBaseError):
    """Base class for service-level errors."""


class ServiceNotInitializedError(ServiceError):
    """Raised when a service is not properly initialized."""

    def __init__(self, service_name: str) -> None:
        message = f"{service_name} service is not initialized"
        super().__init__(message, error_code="SERVICE_NOT_INITIALIZED")
        self.service_name = service_name


class ServiceUnavailableError(ServiceError):
    """Raised when a required service is unavailable."""

    def __init__(self, service_name: str, reason: str | None = None) -> None:
        message = f"{service_name} service is unavailable"
        if reason:
            message += f": {reason}"
        super().__init__(message, error_code="SERVICE_UNAVAILABLE")
        self.service_name = service_name
        self.reason = reason


# ==============================================================================
# Authentication and Authorization Exceptions
# ==============================================================================


class AuthenticationError(ClarityBaseError):
    """Base class for authentication errors."""

    def __init__(self, message: str, **kwargs: dict[str, Any]) -> None:
        super().__init__(message, error_code="AUTHENTICATION_ERROR", **kwargs)


class UserNotFoundError(AuthenticationError):
    """Raised when user is not found."""


class InvalidCredentialsError(AuthenticationError):
    """Raised when credentials are invalid."""

    def __init__(self, message: str, **kwargs: dict[str, Any]) -> None:
        """Initialize with specific error code."""
        super().__init__(message, **kwargs)
        self.error_code = "INVALID_CREDENTIALS"


class UserAlreadyExistsError(AuthenticationError):
    """Raised when user already exists."""


class EmailNotVerifiedError(AuthenticationError):
    """Raised when email is not verified."""


class AuthorizationError(ClarityBaseError):
    """Base class for authorization errors."""

    def __init__(self, message: str, **kwargs: dict[str, Any]) -> None:
        super().__init__(message, error_code="AUTHORIZATION_ERROR", **kwargs)


class AccountDisabledError(AuthenticationError):
    """Raised when user account is disabled."""

    def __init__(self, user_id: str) -> None:
        message = f"User account {user_id} is disabled"
        super().__init__(message)
        self.user_id = user_id


class AccessDeniedError(AuthorizationError):
    """Raised when access to a resource is denied."""

    def __init__(self, resource: str, user_id: str | None = None) -> None:
        message = f"Access denied to {resource}"
        if user_id:
            message += f" for user {user_id}"
        super().__init__(message)
        self.resource = resource
        self.user_id = user_id


# ==============================================================================
# Configuration and Setup Exceptions
# ==============================================================================


class ConfigurationError(ClarityBaseError):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self, message: str, *, config_key: str | None = None, **kwargs: dict[str, Any]
    ) -> None:
        super().__init__(message, error_code="CONFIGURATION_ERROR", **kwargs)
        self.config_key = config_key


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(self, config_key: str) -> None:
        message = f"Missing required configuration: {config_key}"
        super().__init__(message, config_key=config_key)


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration value is invalid."""

    def __init__(
        self, config_key: str, value: object, reason: str | None = None
    ) -> None:
        message = f"Invalid configuration value for {config_key}: {value}"
        if reason:
            message += f" ({reason})"
        super().__init__(message, config_key=config_key)
        self.value = value
        self.reason = reason


# ==============================================================================
# Cache and Performance Exceptions
# ==============================================================================


class CacheError(ClarityBaseError):
    """Base class for cache-related errors."""

    def __init__(self, message: str, **kwargs: dict[str, Any]) -> None:
        super().__init__(message, error_code="CACHE_ERROR", **kwargs)


class CacheKeyError(CacheError):
    """Raised when cache key operations fail."""

    def __init__(self, cache_key: str, operation: str) -> None:
        message = f"Cache {operation} failed for key: {cache_key}"
        super().__init__(message)
        self.cache_key = cache_key
        self.operation = operation


# ==============================================================================
# Utility Functions for Exception Creation
# ==============================================================================


def create_validation_error(
    field_name: str, expected_type: str, actual_value: object
) -> DataValidationError:
    """Create a standardized validation error for type mismatches."""
    actual_type = type(actual_value).__name__
    message = f"Field '{field_name}' expected {expected_type}, got {actual_type}: {actual_value}"
    return DataValidationError(message, field_name=field_name)


def create_numeric_validation_error(
    field_name: str, value: object
) -> InvalidNHANESStatsDataError:
    """Create a validation error for non-numeric values where numbers are expected."""
    actual_type = type(value).__name__
    return InvalidNHANESStatsDataError(
        data_type=field_name,
        expected_type="numeric (int or float)",
        actual_type=actual_type,
    )
