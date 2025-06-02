"""Custom exception hierarchy for the Clarity Digital Twin platform.

This module provides a comprehensive exception hierarchy following clean code principles,
enabling precise error handling and meaningful error messages throughout the application.

The exception hierarchy is designed to be:
- Specific and descriptive
- Easy to catch and handle at appropriate levels
- Consistent in structure and naming
- Self-documenting through clear names and messages
"""

from typing import Any


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
        self, message: str, *, field_name: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(message, error_code="DATA_VALIDATION_ERROR", **kwargs)
        self.field_name = field_name


class InvalidStepCountDataError(DataValidationError):
    """Raised when step count data is invalid or malformed."""


class InvalidNHANESStatsError(DataValidationError):
    """Raised when NHANES statistics data is invalid or malformed."""


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
        self, message: str, *, request_id: str | None = None, **kwargs: Any
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

    def __init__(self, message: str, **kwargs: Any) -> None:
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

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_code="AUTHENTICATION_ERROR", **kwargs)


class AuthorizationError(ClarityBaseError):
    """Base class for authorization errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
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
        self, message: str, *, config_key: str | None = None, **kwargs: Any
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

    def __init__(self, config_key: str, value: Any, reason: str | None = None) -> None:
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

    def __init__(self, message: str, **kwargs: Any) -> None:
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
    field_name: str, expected_type: str, actual_value: Any
) -> DataValidationError:
    """Create a standardized validation error for type mismatches."""
    actual_type = type(actual_value).__name__
    message = f"Field '{field_name}' expected {expected_type}, got {actual_type}: {actual_value}"
    return DataValidationError(message, field_name=field_name)


def create_numeric_validation_error(
    field_name: str, value: Any
) -> InvalidNHANESStatsDataError:
    """Create a validation error for non-numeric values where numbers are expected."""
    actual_type = type(value).__name__
    return InvalidNHANESStatsDataError(
        data_type=field_name,
        expected_type="numeric (int or float)",
        actual_type=actual_type,
    )
