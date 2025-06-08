"""User models for the CLARITY Digital Twin Platform."""

from datetime import datetime
from typing import Any, ClassVar

from pydantic import BaseModel, EmailStr, Field, ConfigDict


class UserBase(BaseModel):
    """Base user model for common user fields."""

    email: EmailStr = Field(..., description="User's email address")
    full_name: str | None = Field(None, description="User's full name", max_length=100)
    is_active: bool = Field(True, description="Flag for active user accounts")

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


class User(UserBase):
    """User model for authenticated users."""

    uid: str = Field(..., description="Firebase user ID")
    display_name: str | None = Field(None, description="User display name")
    email_verified: bool = Field(default=False, description="Whether email is verified")
    firebase_token: str | None = Field(None, description="Firebase ID token")
    firebase_token_exp: float | None = Field(
        None, description="Firebase ID token expiration timestamp (Unix)"
    )
    created_at: datetime | None = Field(None, description="Account creation timestamp")
    last_login: datetime | None = Field(None, description="Last login timestamp")
    profile: dict[str, Any] | None = Field(
        None, description="Additional user profile data"
    )

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )


class UserProfile(BaseModel):
    """Extended user profile information."""

    uid: str = Field(..., description="Firebase user ID")
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
                "uid": "firebase_user_id_123",
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

    uid: str = Field(..., description="Firebase user ID")
    session_id: str = Field(..., description="Session identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = Field(None)
    device_info: dict[str, Any] | None = Field(None, description="Device information")
    ip_address: str | None = Field(None, description="Client IP address")
    user_agent: str | None = Field(None, description="Client user agent")

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )
