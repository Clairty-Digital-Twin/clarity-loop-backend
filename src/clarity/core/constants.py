"""Core constants for the Clarity Digital Twin platform.

This module centralizes all magic numbers, thresholds, and configuration values
following the Single Responsibility Principle and making the codebase more maintainable.

Constants are organized by domain and feature area for clarity and discoverability.
"""

# removed - breaks FastAPI

from typing import Final

# ==============================================================================
# Time and Duration Constants
# ==============================================================================

# Time periods in minutes
MINUTES_PER_HOUR: Final[int] = 60
MINUTES_PER_DAY: Final[int] = 1440  # 24 * 60
MINUTES_PER_WEEK: Final[int] = 10080  # 7 * 24 * 60

# Time periods in seconds
SECONDS_PER_MINUTE: Final[int] = 60
SECONDS_PER_HOUR: Final[int] = 3600  # 60 * 60
CACHE_TTL_DEFAULT_SECONDS: Final[int] = 300  # 5 minutes

# ==============================================================================
# Health Data and Actigraphy Constants
# ==============================================================================

# Step count thresholds
MAX_REALISTIC_STEPS_PER_MINUTE: Final[int] = 1000
MIN_ACTIVITY_LEVEL: Final[float] = 0.5
SQRT_MIN_ACTIVITY: Final[float] = 0.5  # sqrt(0.5) for minimal activity periods

# Proxy actigraphy transformation
CIRCADIAN_PADDING_VARIATION_STD: Final[float] = 0.1
CIRCADIAN_PATTERN_AMPLITUDE: Final[float] = 0.2
CIRCADIAN_DECAY_FACTOR: Final[float] = -0.5
SMOOTHING_WINDOW_SIZE: Final[int] = 5

# Quality scoring thresholds
EXTREME_VALUE_LOWER_THRESHOLD: Final[float] = -3.0
EXTREME_VALUE_UPPER_THRESHOLD: Final[float] = 3.0
PROXY_VALUE_CLIP_MIN: Final[float] = -4.0
PROXY_VALUE_CLIP_MAX: Final[float] = 4.0

# Data quality penalties and weights
ZERO_PERCENTAGE_PENALTY_FACTOR: Final[float] = 1.5
EXTREME_VALUE_PENALTY_FACTOR: Final[float] = 2.0
PADDING_PENALTY_FACTOR: Final[float] = 0.5

# Quality score weights (must sum to 1.0)
QUALITY_WEIGHT_COMPLETENESS: Final[float] = 0.3
QUALITY_WEIGHT_VARIABILITY: Final[float] = 0.25
QUALITY_WEIGHT_REALISTIC: Final[float] = 0.2
QUALITY_WEIGHT_PROXY_SIGNAL: Final[float] = 0.15
QUALITY_WEIGHT_PADDING: Final[float] = 0.1

# ==============================================================================
# ML Model and Inference Constants
# ==============================================================================

# Inference engine settings
DEFAULT_BATCH_SIZE: Final[int] = 4
DEFAULT_BATCH_TIMEOUT_MS: Final[int] = 100
DEFAULT_INFERENCE_TIMEOUT_SECONDS: Final[float] = 30.0
BATCH_PROCESSOR_ERROR_SLEEP_SECONDS: Final[float] = 0.1

# Performance monitoring
PERFORMANCE_TIMEOUT_WARNING_THRESHOLD_MS: Final[float] = 1000.0
CACHE_CLEANUP_BATCH_SIZE: Final[int] = 100

# ==============================================================================
# API and HTTP Constants
# ==============================================================================

# HTTP timeouts
DEFAULT_REQUEST_TIMEOUT_SECONDS: Final[float] = 30.0
FASTAPI_DEPENDENCY_INJECTION_TIMEOUT: Final[float] = 1.0

# Pagination defaults
DEFAULT_PAGE_LIMIT: Final[int] = 10
DEFAULT_PAGE_OFFSET: Final[int] = 0
MAX_PAGE_LIMIT: Final[int] = 100

# HTTP Status codes for common responses
HTTP_STATUS_OK: Final[int] = 200
HTTP_STATUS_CREATED: Final[int] = 201
HTTP_STATUS_NO_CONTENT: Final[int] = 204
HTTP_STATUS_BAD_REQUEST: Final[int] = 400
HTTP_STATUS_UNAUTHORIZED: Final[int] = 401
HTTP_STATUS_FORBIDDEN: Final[int] = 403
HTTP_STATUS_NOT_FOUND: Final[int] = 404
HTTP_STATUS_UNPROCESSABLE_ENTITY: Final[int] = 422
HTTP_STATUS_INTERNAL_SERVER_ERROR: Final[int] = 500

# ==============================================================================
# NHANES Statistics Constants
# ==============================================================================

# NHANES reference data
NHANES_DEFAULT_YEAR: Final[int] = 2025
NHANES_FALLBACK_YEAR: Final[int] = 2024

# Statistical thresholds
EXTREME_OUTLIER_THRESHOLD: Final[float] = 3.0
Z_SCORE_NORMALIZATION_EPSILON: Final[float] = 1e-6

# ==============================================================================
# Security and Hashing Constants
# ==============================================================================

# Secure hashing
HASH_ALGORITHM: Final[str] = "sha256"
CACHE_KEY_TRUNCATION_LENGTH: Final[int] = 16

# Authentication constants
AUTH_HEADER_TYPE_BEARER: Final[str] = (
    "bearer"  # nosec B105 - not a password, just a token type
)
AUTH_TOKEN_DEFAULT_EXPIRY_SECONDS: Final[int] = 3600
AUTH_SCOPE_FULL_ACCESS: Final[str] = "full_access"

# ==============================================================================
# Error Handling Constants
# ==============================================================================

# Error codes
ERROR_CODE_VALIDATION_FAILED: Final[str] = "VALIDATION_FAILED"
ERROR_CODE_ACCOUNT_DISABLED: Final[str] = "ACCOUNT_DISABLED"
ERROR_CODE_ACCESS_DENIED: Final[str] = "ACCESS_DENIED"
ERROR_CODE_INSIGHT_GENERATION_FAILED: Final[str] = "INSIGHT_GENERATION_FAILED"
ERROR_CODE_SERVICE_UNAVAILABLE: Final[str] = "SERVICE_UNAVAILABLE"

# Retry and backoff
DEFAULT_RETRY_ATTEMPTS: Final[int] = 3
EXPONENTIAL_BACKOFF_BASE_SECONDS: Final[float] = 0.1
MAX_BACKOFF_SECONDS: Final[float] = 5.0

# ==============================================================================
# Development and Testing Constants
# ==============================================================================

# Test data generation
TEST_WEEK_SAMPLE_SIZE: Final[int] = MINUTES_PER_WEEK
TEST_QUALITY_THRESHOLD: Final[float] = 0.8
PERFORMANCE_TEST_ITERATIONS: Final[int] = 1000

# ==============================================================================
# Configuration and Environment Constants
# ==============================================================================

# Development flags
DEV_PROJECT_ID: Final[str] = "dev-project"
DEVELOPMENT_MODE_LOG_LEVEL: Final[str] = "DEBUG"
PRODUCTION_MODE_LOG_LEVEL: Final[str] = "INFO"

# Service metadata
SERVICE_VERSION: Final[str] = "1.0.0"
API_VERSION: Final[str] = "v1"

# AWS Service Limits
S3_LIFECYCLE_TRANSITION_IA_DAYS: Final[int] = 30
S3_LIFECYCLE_TRANSITION_GLACIER_DAYS: Final[int] = 90
S3_LIFECYCLE_EXPIRATION_DAYS: Final[int] = 365
DYNAMODB_BATCH_WRITE_ITEM_LIMIT: Final[int] = 25
COGNITO_PASSWORD_MIN_LENGTH: Final[int] = 8

# ==============================================================================
# Validation Constants
# ==============================================================================

# Data validation
MIN_STEP_COUNT_LIST_LENGTH: Final[int] = 1
MIN_QUALITY_SCORE: Final[float] = 0.0
MAX_QUALITY_SCORE: Final[float] = 1.0

# Input validation
MAX_USERNAME_LENGTH: Final[int] = 255
MAX_REQUEST_ID_LENGTH: Final[int] = 64
MAX_ERROR_MESSAGE_LENGTH: Final[int] = 1000

# JWT and Bearer token validation
BEARER_TOKEN_PARTS_COUNT: Final[int] = 2
JWT_TOKEN_PARTS_COUNT: Final[int] = 3
