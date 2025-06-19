"""Type definitions for the Clarity Digital Twin platform.

This module provides comprehensive type definitions to ensure type safety
throughout the application, replacing generic Any types with specific,
meaningful type annotations.

Types are organized by domain and usage patterns for better discoverability
and maintainability.
"""

# removed - breaks FastAPI

from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

# ==============================================================================
# Generic Type Variables
# ==============================================================================

T = TypeVar("T")
ModelT = TypeVar("ModelT", bound=BaseModel)
ExceptionT = TypeVar("ExceptionT", bound=Exception)

# ==============================================================================
# Numeric and Array Types
# ==============================================================================

# Numpy array types
FloatArray: TypeAlias = NDArray[np.floating[Any]]
BoolArray: TypeAlias = NDArray[np.bool_]
IntArray: TypeAlias = NDArray[np.integer[Any]]
NumericArray: TypeAlias = FloatArray | IntArray

# Scalar numeric types
NumericValue: TypeAlias = int | float | np.integer[Any] | np.floating[Any]
StepCount: TypeAlias = float
QualityScore: TypeAlias = float  # Always between 0.0 and 1.0
ZScore: TypeAlias = float
ProxyValue: TypeAlias = float

# ==============================================================================
# Configuration and Settings Types
# ==============================================================================

# Configuration dictionaries
ConfigDict: TypeAlias = dict[str, Any]
NestedConfigDict: TypeAlias = dict[str, str | int | float | bool | dict[str, Any]]
EnvironmentVariables: TypeAlias = dict[str, str]

# Service configuration
ServiceConfig: TypeAlias = Mapping[str, str | int | float | bool]
ModelConfig: TypeAlias = dict[str, str | int | float]

# ==============================================================================
# API and Request/Response Types
# ==============================================================================

# HTTP related types
HTTPHeaders: TypeAlias = dict[str, str]
QueryParams: TypeAlias = dict[str, str | int | float | bool]
RouteParams: TypeAlias = dict[str, str]

# Request/Response data
RequestData: TypeAlias = dict[str, Any]
ResponseData: TypeAlias = dict[str, Any]
ErrorDetails: TypeAlias = dict[str, str | int | float | bool]
MetadataDict: TypeAlias = dict[str, str | int | float | datetime]

# Pagination
PaginationParams: TypeAlias = dict[str, int]  # limit, offset, etc.

# ==============================================================================
# Health Data and Medical Types
# ==============================================================================

# Step count and activity data
StepCountList: TypeAlias = list[StepCount]
TimestampList: TypeAlias = list[datetime]
ActivityData: TypeAlias = dict[str, StepCountList | TimestampList]

# User demographics and metadata
UserMetadata: TypeAlias = dict[str, str | int | float | bool]
Demographics: TypeAlias = dict[str, str | int]  # age, sex, etc.

# NHANES statistics
NHANESStats: TypeAlias = dict[str, float | str | int]
NHANESYearData: TypeAlias = dict[str, NHANESStats]
StratifiedStats: TypeAlias = dict[str, NHANESStats]

# ==============================================================================
# ML Model and Inference Types
# ==============================================================================

# Model input/output
ModelInput: TypeAlias = dict[str, FloatArray | str | float]
ModelOutput: TypeAlias = dict[str, float | str | list[float]]
PredictionResult: TypeAlias = dict[str, float | str | int]

# Analysis results
AnalysisMetrics: TypeAlias = dict[str, NumericValue]
TransformationStats: TypeAlias = dict[str, int | float | dict[str, float]]
QualityMetrics: TypeAlias = dict[str, QualityScore]

# Caching
CacheKey: TypeAlias = str
CachedValue: TypeAlias = tuple[Any, float]  # (value, timestamp)
CacheStorage: TypeAlias = dict[CacheKey, CachedValue]

# ==============================================================================
# Protocol Definitions for Dependency Injection
# ==============================================================================


class AsyncCallable(Protocol):
    """Protocol for async callable objects."""

    async def __call__(self, *args: object, **kwargs: object) -> object:
        """Async call method."""
        ...


class CacheProvider(Protocol):
    """Protocol for cache implementations."""

    async def get(self, key: str) -> object | None:
        """Get value from cache."""
        ...

    async def set(self, key: str, value: object) -> None:
        """Set value in cache."""
        ...

    def clear(self) -> None:
        """Clear cache."""
        ...


class ConfigProvider(Protocol):
    """Protocol for configuration providers."""

    def get(self, key: str, default: T | None = None) -> T | None:
        """Get configuration value."""
        ...

    def is_development(self) -> bool:
        """Check if in development mode."""
        ...


class LoggerProtocol(Protocol):
    """Protocol for logger objects."""

    def debug(self, msg: str, *args: object) -> None: ...
    def info(self, msg: str, *args: object) -> None: ...
    def warning(self, msg: str, *args: object) -> None: ...
    def error(self, msg: str, *args: object) -> None: ...
    def exception(self, msg: str, *args: object) -> None: ...


# ==============================================================================
# Function Type Aliases
# ==============================================================================

# Error handlers
ErrorHandler: TypeAlias = Callable[[Exception], Any]
AsyncErrorHandler: TypeAlias = Callable[[Exception], Awaitable[Any]]

# Data validators
DataValidator: TypeAlias = Callable[[Any], bool]
AsyncDataValidator: TypeAlias = Callable[[Any], Awaitable[bool]]

# Transformation functions
DataTransformer: TypeAlias = Callable[[Any], Any]
AsyncDataTransformer: TypeAlias = Callable[[Any], Awaitable[Any]]

# Event handlers
EventHandler: TypeAlias = Callable[..., None]
AsyncEventHandler: TypeAlias = Callable[..., Awaitable[None]]

# ==============================================================================
# Data Processing Types
# ==============================================================================

# Time series data
TimeSeriesPoint: TypeAlias = tuple[datetime, NumericValue]
TimeSeries: TypeAlias = Sequence[TimeSeriesPoint]
TimeSeriesDict: TypeAlias = dict[str, TimeSeries]

# Statistical data
StatisticalSummary: TypeAlias = dict[str, NumericValue]  # mean, std, etc.
DistributionParams: TypeAlias = dict[str, NumericValue]  # parameters for distributions

# Validation results
ValidationResult: TypeAlias = dict[str, bool | str | int | float]
QualityAssessment: TypeAlias = dict[str, QualityScore | bool | str]

# ==============================================================================
# Specialized Domain Types
# ==============================================================================

# Authentication and authorization
UserID: TypeAlias = str
SessionToken: TypeAlias = str
Permission: TypeAlias = str
UserPermissions: TypeAlias = set[Permission]

# Request tracking
RequestID: TypeAlias = str
CorrelationID: TypeAlias = str
TraceID: TypeAlias = str

# Service identification
ServiceName: TypeAlias = str
ServiceVersion: TypeAlias = str
ServiceEndpoint: TypeAlias = str

# Data integrity
Checksum: TypeAlias = str
DataHash: TypeAlias = str
IntegrityToken: TypeAlias = str

# ==============================================================================
# Composite Types for Complex Structures
# ==============================================================================

# Complete user context
UserContext: TypeAlias = dict[str, UserID | bool | UserPermissions | UserMetadata]

# Complete request context
RequestContext: TypeAlias = dict[str, RequestID | UserID | datetime | HTTPHeaders]

# Service health status
ServiceHealth: TypeAlias = dict[str, bool | str | datetime | dict[str, Any]]

# Complete analysis result
AnalysisResult: TypeAlias = dict[
    str,
    str | float | datetime | AnalysisMetrics | QualityAssessment | TransformationStats,
]

# ==============================================================================
# Type Guards and Validation Helpers
# ==============================================================================


def is_numeric_value(value: object) -> bool:
    """Type guard for numeric values."""
    return isinstance(value, (int, float, np.integer, np.floating))


def is_step_count_list(value: object) -> bool:
    """Type guard for step count lists."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(item, (int, float)) and item >= 0 for item in value)
    )


def is_quality_score(value: object) -> bool:
    """Type guard for quality scores (0.0 to 1.0)."""
    return isinstance(value, (int, float)) and 0.0 <= value <= 1.0


def is_timestamp_list(value: object) -> bool:
    """Type guard for timestamp lists."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(item, datetime) for item in value)
    )
