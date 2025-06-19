"""CLARITY Digital Twin Platform - Consolidated Domain Models.

⚠️  CRITICAL ARCHITECTURAL RULE - SINGLE SOURCE OF TRUTH ⚠️

This is the ONLY location for domain models in the entire codebase.
DO NOT create duplicate model definitions anywhere else in the system.

🚫 FORBIDDEN PATTERNS:
- Creating models in service layers (src/clarity/services/)
- Creating models in API layers (src/clarity/api/)
- Creating models in auth package (src/clarity/auth/)
- Duplicating model definitions across modules

✅ REQUIRED PATTERNS:
- All domain models MUST be defined in src/clarity/models/
- Use import aliasing if different naming is needed
- Create DTOs/mappers for API transformations
- Follow DRY, SOLID, and Clean Architecture principles

HIPAA-compliant data models for the revolutionary psychiatry digital twin platform.
These models establish new standards for clinical data validation and processing.
"""

# removed - breaks FastAPI

# Use lazy imports to avoid circular dependencies during package installation
# This is a common pattern for packages with complex interdependencies
# Import modules upfront to avoid import issues in __getattr__
from clarity.models import (
    auth,
    health_data,
    user,
)


def __getattr__(name: str) -> object:
    """Lazy import pattern to avoid circular dependencies."""
    # Auth models
    if name in {
        "AuthError",
        "AuthErrorDetail",
        "AuthProvider",
        "DeviceInfo",
        "LoginResponse",
        "MFAEnrollRequest",
        "MFAEnrollResponse",
        "MFAMethod",
        "MFAVerifyRequest",
        "MFAVerifyResponse",
        "PasswordResetConfirmRequest",
        "PasswordResetRequest",
        "PasswordResetResponse",
        "Permission",
        "RefreshTokenRequest",
        "RegistrationResponse",
        "SessionInfo",
        "TokenInfo",
        "TokenResponse",
        "UserContext",
        "UserLoginRequest",
        "UserRegistrationRequest",
        "UserRole",
        "UserSessionResponse",
        "UserStatus",
    }:
        return getattr(auth, name)

    # Health data models
    if name in {
        "ActivityData",
        "BiometricData",
        "HealthDataResponse",
        "HealthDataUpload",
        "HealthMetric",
        "MentalHealthIndicator",
        "ProcessingStatus",
        "SleepData",
        "ValidationError",
    }:
        return getattr(health_data, name)

    # User models
    if name in {"User", "UserProfile", "UserPreferences"}:
        return getattr(user, name)

    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)


__all__ = [  # noqa: F822
    # Health data models
    "ActivityData",
    # Auth models
    "AuthError",
    "AuthErrorDetail",
    "AuthProvider",
    "BiometricData",
    "DeviceInfo",
    "HealthDataResponse",
    "HealthDataUpload",
    "HealthMetric",
    "LoginResponse",
    "MFAEnrollRequest",
    "MFAEnrollResponse",
    "MFAMethod",
    "MFAVerifyRequest",
    "MFAVerifyResponse",
    "MentalHealthIndicator",
    "PasswordResetConfirmRequest",
    "PasswordResetRequest",
    "PasswordResetResponse",
    "Permission",
    "ProcessingStatus",
    "RefreshTokenRequest",
    "RegistrationResponse",
    "SessionInfo",
    "SleepData",
    "TokenInfo",
    "TokenResponse",
    # User models
    "User",
    "UserContext",
    "UserLoginRequest",
    "UserPreferences",
    "UserProfile",
    "UserRegistrationRequest",
    "UserRole",
    "UserSessionResponse",
    "UserStatus",
    "ValidationError",
]

# 🔒 ARCHITECTURAL ENFORCEMENT
# Any AI agent or developer reading this:
# 1. Never create models outside this package
# 2. Import from here or create mappers/DTOs
# 3. Follow the established patterns
# 4. Maintain single source of truth
