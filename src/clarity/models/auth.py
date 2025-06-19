"""CLARITY Digital Twin Platform - Authentication Models.

Pydantic models for authentication request/response validation.
Supports user registration, login, token management, and role-based access control.
"""

# removed - breaks FastAPI

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator

# Constants for validation
MIN_PASSWORD_LENGTH = 8


class UserStatus(StrEnum):
    """User account status enumeration."""

    ACTIVE = "active"
    DISABLED = "disabled"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"


class AuthProvider(StrEnum):
    """Authentication provider enumeration.

    These string constants define supported authentication providers and are not
    actual passwords or security credentials.
    """

    EMAIL_PASSWORD = "email_password"  # noqa: S105 # nosec: B105 - String literal identifies auth provider type, not a password
    GOOGLE = "google"
    APPLE = "apple"
    MICROSOFT = "microsoft"


class MFAMethod(StrEnum):
    """Multi-factor authentication method enumeration."""

    SMS = "sms"
    EMAIL = "email"
    TOTP = "totp"
    PHONE_CALL = "phone_call"


class UserRole(StrEnum):
    """User roles for access control."""

    PATIENT = "patient"
    CLINICIAN = "clinician"
    RESEARCHER = "researcher"
    ADMIN = "admin"


class Permission(StrEnum):
    """Permission types for granular access control."""

    READ_OWN_DATA = "read_own_data"
    WRITE_OWN_DATA = "write_own_data"
    READ_PATIENT_DATA = "read_patient_data"
    WRITE_PATIENT_DATA = "write_patient_data"
    READ_ANONYMIZED_DATA = "read_anonymized_data"
    MANAGE_USERS = "manage_users"
    SYSTEM_ADMIN = "system_admin"


# Core Domain Models


class UserContext(BaseModel):
    """User context containing authentication and authorization information."""

    user_id: str = Field(description="Unique user identifier")
    email: str | None = Field(None, description="User email address")
    role: UserRole = Field(UserRole.PATIENT, description="User role")
    permissions: list[Permission] = Field(
        default_factory=list, description="User permissions"
    )
    is_verified: bool = Field(default=False, description="Email verification status")
    is_active: bool = Field(default=True, description="User account status")
    custom_claims: dict[str, Any] = Field(
        default_factory=dict, description="Custom AWS Cognito claims"
    )
    created_at: datetime | None = Field(None, description="Account creation timestamp")
    last_login: datetime | None = Field(None, description="Last login timestamp")


@dataclass
class TokenInfo:
    """Token information from AWS Cognito."""

    token: str = ""
    expires_at: float = 0.0
    user_id: str = ""
    email: str = ""
    cognito_claims: dict[str, Any] = field(default_factory=dict)


class AuthError(Exception):
    """Authentication and authorization error exception."""

    def __init__(
        self, message: str, status_code: int = 401, error_code: str = "auth_error"
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.message)


# Request Models


class UserRegistrationRequest(BaseModel):
    """Request model for user registration."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(
        ..., min_length=MIN_PASSWORD_LENGTH, max_length=128, description="User password"
    )
    first_name: str = Field(
        ..., min_length=1, max_length=50, description="User first name"
    )
    last_name: str = Field(
        ..., min_length=1, max_length=50, description="User last name"
    )
    phone_number: str | None = Field(None, description="Optional phone number for MFA")
    terms_accepted: bool = Field(..., description="Terms and conditions acceptance")
    privacy_policy_accepted: bool = Field(..., description="Privacy policy acceptance")

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password complexity."""
        if len(v) < MIN_PASSWORD_LENGTH:
            msg = f"Password must be at least {MIN_PASSWORD_LENGTH} characters long"
            raise ValueError(msg)

        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v)

        if not (has_upper and has_lower and has_digit and has_special):
            msg = (
                "Password must contain at least one uppercase letter, "
                "one lowercase letter, one digit, and one special character"
            )
            raise ValueError(msg)

        return v

    @field_validator("terms_accepted", "privacy_policy_accepted")
    @classmethod
    def validate_acceptance(cls, v: bool) -> bool:  # noqa: FBT001
        """Validate that terms and privacy policy are accepted."""
        if not v:
            msg = "Terms and privacy policy must be accepted"
            raise ValueError(msg)
        return v


class UserLoginRequest(BaseModel):
    """Request model for user login."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")
    remember_me: bool = Field(
        default=False, description="Remember user for extended session"
    )
    device_info: dict[str, Any] | None = Field(
        None, description="Optional device information for security tracking"
    )


class RefreshTokenRequest(BaseModel):
    """Request model for token refresh."""

    refresh_token: str = Field(..., description="Refresh token")


class PasswordResetRequest(BaseModel):
    """Request model for password reset initiation."""

    email: EmailStr = Field(..., description="User email address")


class PasswordResetConfirmRequest(BaseModel):
    """Request model for password reset confirmation."""

    token: str = Field(..., description="Password reset token")
    new_password: str = Field(
        ..., min_length=MIN_PASSWORD_LENGTH, max_length=128, description="New password"
    )

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password complexity."""
        if len(v) < MIN_PASSWORD_LENGTH:
            msg = f"Password must be at least {MIN_PASSWORD_LENGTH} characters long"
            raise ValueError(msg)

        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v)

        if not (has_upper and has_lower and has_digit and has_special):
            msg = (
                "Password must contain at least one uppercase letter, "
                "one lowercase letter, one digit, and one special character"
            )
            raise ValueError(msg)

        return v


class MFAEnrollRequest(BaseModel):
    """Request model for MFA enrollment."""

    method: MFAMethod = Field(..., description="MFA method to enroll")
    phone_number: str | None = Field(None, description="Phone number for SMS/call MFA")


class MFAVerifyRequest(BaseModel):
    """Request model for MFA verification."""

    method: MFAMethod = Field(..., description="MFA method being verified")
    verification_code: str = Field(..., description="MFA verification code")
    session_token: str = Field(
        ..., description="Temporary session token from initial auth"
    )


# Response Models


class TokenResponse(BaseModel):
    """Response model for token-based authentication."""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: Literal["bearer"] = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    scope: str | None = Field(None, description="Token scope")


class UserSessionResponse(BaseModel):
    """Response model for user session information."""

    user_id: UUID = Field(..., description="User unique identifier")
    email: EmailStr = Field(..., description="User email address")
    first_name: str = Field(..., description="User first name")
    last_name: str = Field(..., description="User last name")
    role: str = Field(..., description="User role")
    permissions: list[str] = Field(..., description="User permissions")
    status: UserStatus = Field(..., description="User account status")
    last_login: datetime | None = Field(None, description="Last login timestamp")
    mfa_enabled: bool = Field(..., description="Whether MFA is enabled")
    email_verified: bool = Field(..., description="Whether email is verified")
    created_at: datetime = Field(..., description="Account creation timestamp")


class LoginResponse(BaseModel):
    """Response model for successful login."""

    user: UserSessionResponse = Field(..., description="User session information")
    tokens: TokenResponse = Field(..., description="Authentication tokens")
    requires_mfa: bool = Field(
        default=False, description="Whether MFA verification is required"
    )
    mfa_session_token: str | None = Field(
        None, description="Temporary session token for MFA completion"
    )


class RegistrationResponse(BaseModel):
    """Response model for user registration."""

    user_id: UUID = Field(..., description="Created user unique identifier")
    email: EmailStr = Field(..., description="User email address")
    status: UserStatus = Field(..., description="User account status")
    verification_email_sent: bool = Field(
        ..., description="Whether verification email was sent"
    )
    created_at: datetime = Field(..., description="Account creation timestamp")


class PasswordResetResponse(BaseModel):
    """Response model for password reset initiation."""

    message: str = Field(..., description="Success message")
    reset_token_sent: bool = Field(..., description="Whether reset token was sent")


class MFAEnrollResponse(BaseModel):
    """Response model for MFA enrollment."""

    method: MFAMethod = Field(..., description="Enrolled MFA method")
    secret_key: str | None = Field(
        None, description="TOTP secret key (for authenticator apps)"
    )
    qr_code_url: str | None = Field(None, description="QR code URL for TOTP setup")
    backup_codes: list[str] | None = Field(None, description="Backup recovery codes")


class MFAVerifyResponse(BaseModel):
    """Response model for MFA verification."""

    verified: bool = Field(..., description="Whether MFA verification was successful")
    tokens: TokenResponse | None = Field(
        None, description="Authentication tokens (if successful)"
    )
    user: UserSessionResponse | None = Field(
        None, description="User session (if successful)"
    )


# Error Models


class AuthErrorDetail(BaseModel):
    """Detailed error information for authentication failures."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    field: str | None = Field(None, description="Field that caused the error")


class AuthErrorResponse(BaseModel):
    """Authentication error response."""

    error: str = Field(..., description="Error type")
    error_description: str = Field(..., description="Detailed error description")
    error_details: list[AuthErrorDetail] | None = Field(
        None, description="Additional error details"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# Utility Models


class DeviceInfo(BaseModel):
    """Device information for security tracking."""

    device_id: str | None = Field(None, description="Unique device identifier")
    device_type: str | None = Field(
        None, description="Device type (mobile, desktop, etc.)"
    )
    os: str | None = Field(None, description="Operating system")
    browser: str | None = Field(None, description="Browser information")
    ip_address: str | None = Field(None, description="IP address")
    user_agent: str | None = Field(None, description="User agent string")


class SessionInfo(BaseModel):
    """Session metadata for security tracking."""

    session_id: str = Field(..., description="Session identifier")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    expires_at: datetime = Field(..., description="Session expiration time")
    device_info: DeviceInfo | None = Field(None, description="Device information")
    ip_address: str | None = Field(None, description="Session IP address")
