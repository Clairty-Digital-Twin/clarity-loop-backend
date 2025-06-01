"""CLARITY Digital Twin Platform - Authentication Models.

Data models for authentication and authorization.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


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
        default_factory=dict, description="Custom Firebase claims"
    )
    created_at: datetime | None = Field(None, description="Account creation timestamp")
    last_login: datetime | None = Field(None, description="Last login timestamp")


class AuthError(Exception):
    """Authentication and authorization error."""

    def __init__(
        self, message: str, status_code: int = 401, error_code: str = "auth_error"
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.message)


class TokenInfo(BaseModel):
    """Token information from Firebase."""

    token: str = Field(description="JWT token")
    user_id: str = Field(description="User ID from token")
    email: str | None = Field(None, description="Email from token")
    issued_at: datetime = Field(description="Token issued timestamp")
    expires_at: datetime = Field(description="Token expiration timestamp")
    is_admin: bool = Field(default=False, description="Admin status from custom claims")
    custom_claims: dict[str, Any] = Field(
        default_factory=dict, description="Custom claims from token"
    )
