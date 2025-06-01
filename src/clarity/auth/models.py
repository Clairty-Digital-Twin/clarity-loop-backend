"""CLARITY Digital Twin Platform - Authentication Models

Data models for authentication and authorization.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UserRole(str, Enum):
    """User roles for access control."""
    PATIENT = "patient"
    CLINICIAN = "clinician"
    RESEARCHER = "researcher"
    ADMIN = "admin"


class Permission(str, Enum):
    """Permission types for granular access control."""
    READ_OWN_DATA = "read_own_data"
    WRITE_OWN_DATA = "write_own_data"
    READ_PATIENT_DATA = "read_patient_data"
    WRITE_PATIENT_DATA = "write_patient_data"
    READ_ANONYMIZED_DATA = "read_anonymized_data"
    MANAGE_USERS = "manage_users"
    SYSTEM_ADMIN = "system_admin"


class UserContext(BaseModel):
    """User context extracted from Firebase token."""
    user_id: str = Field(..., description="Firebase user ID")
    email: str | None = Field(None, description="User email address")
    role: UserRole = Field(UserRole.PATIENT, description="User role")
    permissions: list[Permission] = Field(default_factory=list, description="User permissions")
    is_verified: bool = Field(False, description="Email verification status")
    is_active: bool = Field(True, description="User account status")
    custom_claims: dict[str, Any] = Field(default_factory=dict, description="Custom Firebase claims")
    created_at: datetime | None = Field(None, description="Account creation timestamp")
    last_login: datetime | None = Field(None, description="Last login timestamp")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class AuthError(Exception):
    """Authentication and authorization error."""

    def __init__(self, message: str, status_code: int = 401, error_code: str = "auth_error"):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.message)


class TokenInfo(BaseModel):
    """Firebase token information."""
    token: str
    user_id: str
    email: str | None = None
    issued_at: datetime
    expires_at: datetime
    is_admin: bool = False
    custom_claims: dict[str, Any] = Field(default_factory=dict)
