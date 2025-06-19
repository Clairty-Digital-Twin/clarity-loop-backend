"""User models for the CLARITY Digital Twin Platform."""

# removed - breaks FastAPI

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserBase(BaseModel):
    """Base user model for common user fields."""

    email: EmailStr = Field(..., description="User's email address")
    full_name: str | None = Field(None, description="User's full name", max_length=100)
    is_active: bool = Field(default=True, description="Flag for active user accounts")

    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "email": "jane.doe@example.com",
                "full_name": "Jane Doe",
                "is_active": True,
            }
        },
    )


class UserCreate(UserBase):
    """User creation model with password field."""

    password: str = Field(..., description="User's password")

    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "email": "jane.doe@example.com",
                "full_name": "Jane Doe",
                "is_active": True,
                "password": "a_secure_password",
            }
        },
    )


class UserUpdate(BaseModel):
    """User update model."""

    email: EmailStr | None = None
    full_name: str | None = None
    password: str | None = None
    is_active: bool | None = None

    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "email": "new.email@example.com",
                "full_name": "Jane Updated Doe",
                "password": "a_new_secure_password",
                "is_active": False,
            }
        },
    )


class User(BaseModel):
    """User model representing authenticated users."""

    uid: str = Field(..., description="User ID")
    email: str = Field(..., description="User email address")
    display_name: str | None = Field(None, description="User display name")
    cognito_token: str | None = Field(None, description="AWS Cognito ID token")
    cognito_token_exp: float | None = Field(
        None, description="AWS Cognito ID token expiration timestamp (Unix)"
    )
    role: str = Field(default="user", description="User role")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: datetime | None = Field(None, description="Last login timestamp")
    is_active: bool = Field(default=True, description="Account active status")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional user metadata"
    )

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )


class UserProfile(BaseModel):
    """Extended user profile information."""

    uid: str = Field(..., description="User ID (AWS Cognito sub)")
    display_name: str | None = Field(None, description="Display name")
    bio: str | None = Field(None, max_length=500, description="User biography")
    avatar_url: str | None = Field(None, description="Avatar image URL")
    timezone: str | None = Field(None, description="User timezone")
    language: str = Field(default="en", description="Preferred language")
    privacy_settings: dict[str, Any] = Field(
        default_factory=dict, description="Privacy preferences"
    )
    notification_settings: dict[str, Any] = Field(
        default_factory=dict, description="Notification preferences"
    )
    health_goals: dict[str, Any] | None = Field(
        None, description="User health goals and targets"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "uid": "cognito_user_id_123",
                "display_name": "John Doe",
                "bio": "Health enthusiast interested in sleep optimization",
                "timezone": "America/New_York",
                "language": "en",
                "privacy_settings": {"data_sharing": True, "public_profile": False},
                "notification_settings": {
                    "email_insights": True,
                    "push_notifications": True,
                },
                "health_goals": {"target_sleep_hours": 8, "target_steps": 10000},
            }
        }
    )


class UserRegistration(BaseModel):
    """Model for user registration data."""

    email: EmailStr = Field(..., description="User email address")
    display_name: str | None = Field(None, min_length=1, max_length=100)
    timezone: str | None = Field(None, description="User timezone")
    language: str = Field(default="en", description="Preferred language")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "user@example.com",
                "display_name": "John Doe",
                "timezone": "America/New_York",
                "language": "en",
            }
        }
    )


class UserSession(BaseModel):
    """Model for user session information."""

    uid: str = Field(..., description="User ID (AWS Cognito sub)")
    session_id: str = Field(..., description="Session identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = Field(None)
    device_info: dict[str, Any] | None = Field(None, description="Device information")
    ip_address: str | None = Field(None, description="Client IP address")
    user_agent: str | None = Field(None, description="Client user agent")

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )


class UserWithContext(BaseModel):
    """User data with request context."""

    uid: str = Field(..., description="User ID")
    email: str = Field(..., description="User email address")
    display_name: str | None = Field(None, description="User display name")
    role: str = Field(default="user", description="User role")
    request_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Request-specific metadata"
    )

    @staticmethod
    def example() -> dict[str, Any]:
        """Return an example UserWithContext for API documentation."""
        return {
            "uid": "user_123",
            "email": "john.doe@example.com",
            "display_name": "John Doe",
            "role": "user",
            "request_metadata": {"ip": "192.168.1.1", "user_agent": "Mozilla/5.0"},
        }


class UserContext(BaseModel):
    """User context for request processing."""

    uid: str = Field(..., description="User ID")
    email: str = Field(..., description="User email address")
    display_name: str | None = Field(None, description="User display name")
    role: str = Field(default="user", description="User role")
    cognito_sub: str | None = Field(None, description="AWS Cognito subject identifier")

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )
