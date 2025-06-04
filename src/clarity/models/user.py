"""User models for the CLARITY Digital Twin Platform."""

from datetime import datetime
from typing import Any, ClassVar

from pydantic import BaseModel, EmailStr, Field


class User(BaseModel):
    """User model for authenticated users."""

    uid: str = Field(..., description="Firebase user ID")
    email: EmailStr | None = Field(None, description="User email address")
    display_name: str | None = Field(None, description="User display name")
    email_verified: bool = Field(default=False, description="Whether email is verified")
    firebase_token: str | None = Field(None, description="Firebase ID token")
    created_at: datetime | None = Field(None, description="Account creation timestamp")
    last_login: datetime | None = Field(None, description="Last login timestamp")
    profile: dict[str, Any] | None = Field(None, description="Additional user profile data")

    class Config:
        """Pydantic configuration."""

        json_encoders: ClassVar[dict[type, Any]] = {
            datetime: lambda v: v.isoformat() if v else None
        }


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
    health_goals: dict[str, Any] | None = Field(None, description="User health goals and targets")

    class Config:
        """Pydantic configuration."""

        schema_extra: ClassVar[dict[str, Any]] = {
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


class UserRegistration(BaseModel):
    """Model for user registration data."""

    email: EmailStr = Field(..., description="User email address")
    display_name: str | None = Field(None, min_length=1, max_length=100)
    timezone: str | None = Field(None, description="User timezone")
    language: str = Field(default="en", description="Preferred language")

    class Config:
        """Pydantic configuration."""

        schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "email": "user@example.com",
                "display_name": "John Doe",
                "timezone": "America/New_York",
                "language": "en",
            }
        }


class UserUpdate(BaseModel):
    """Model for updating user information."""

    display_name: str | None = Field(None, min_length=1, max_length=100)
    bio: str | None = Field(None, max_length=500)
    timezone: str | None = Field(None)
    language: str | None = Field(None)
    privacy_settings: dict[str, Any] | None = Field(None)
    notification_settings: dict[str, Any] | None = Field(None)
    health_goals: dict[str, Any] | None = Field(None)

    class Config:
        """Pydantic configuration."""

        schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "display_name": "John Smith",
                "bio": "Updated bio information",
                "timezone": "America/Los_Angeles",
                "health_goals": {"target_sleep_hours": 7.5, "target_steps": 12000},
            }
        }


class UserSession(BaseModel):
    """Model for user session information."""

    uid: str = Field(..., description="Firebase user ID")
    session_id: str = Field(..., description="Session identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = Field(None)
    device_info: dict | None = Field(None, description="Device information")
    ip_address: str | None = Field(None, description="Client IP address")
    user_agent: str | None = Field(None, description="Client user agent")

    class Config:
        """Pydantic configuration."""

        json_encoders: ClassVar[dict[type, Any]] = {
            datetime: lambda v: v.isoformat() if v else None
        }
