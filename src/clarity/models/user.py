"""User models for the CLARITY Digital Twin Platform."""

from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, EmailStr, Field


class User(BaseModel):
    """User model for authenticated users."""
    
    uid: str = Field(..., description="Firebase user ID")
    email: Optional[EmailStr] = Field(None, description="User email address")
    display_name: Optional[str] = Field(None, description="User display name")
    email_verified: bool = Field(default=False, description="Whether email is verified")
    firebase_token: Optional[str] = Field(None, description="Firebase ID token")
    created_at: Optional[datetime] = Field(None, description="Account creation timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    profile: Optional[Dict] = Field(None, description="Additional user profile data")
    
    class Config:
        """Pydantic configuration."""
        
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class UserProfile(BaseModel):
    """Extended user profile information."""
    
    uid: str = Field(..., description="Firebase user ID")
    display_name: Optional[str] = Field(None, description="Display name")
    bio: Optional[str] = Field(None, max_length=500, description="User biography")
    avatar_url: Optional[str] = Field(None, description="Avatar image URL")
    timezone: Optional[str] = Field(None, description="User timezone")
    language: str = Field(default="en", description="Preferred language")
    privacy_settings: Dict = Field(default_factory=dict, description="Privacy preferences")
    notification_settings: Dict = Field(default_factory=dict, description="Notification preferences")
    health_goals: Optional[Dict] = Field(None, description="User health goals and targets")
    
    class Config:
        """Pydantic configuration."""
        
        schema_extra = {
            "example": {
                "uid": "firebase_user_id_123",
                "display_name": "John Doe",
                "bio": "Health enthusiast interested in sleep optimization",
                "timezone": "America/New_York",
                "language": "en",
                "privacy_settings": {
                    "data_sharing": True,
                    "public_profile": False
                },
                "notification_settings": {
                    "email_insights": True,
                    "push_notifications": True
                },
                "health_goals": {
                    "target_sleep_hours": 8,
                    "target_steps": 10000
                }
            }
        }


class UserRegistration(BaseModel):
    """Model for user registration data."""
    
    email: EmailStr = Field(..., description="User email address")
    display_name: Optional[str] = Field(None, min_length=1, max_length=100)
    timezone: Optional[str] = Field(None, description="User timezone")
    language: str = Field(default="en", description="Preferred language")
    
    class Config:
        """Pydantic configuration."""
        
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "display_name": "John Doe",
                "timezone": "America/New_York",
                "language": "en"
            }
        }


class UserUpdate(BaseModel):
    """Model for updating user information."""
    
    display_name: Optional[str] = Field(None, min_length=1, max_length=100)
    bio: Optional[str] = Field(None, max_length=500)
    timezone: Optional[str] = Field(None)
    language: Optional[str] = Field(None)
    privacy_settings: Optional[Dict] = Field(None)
    notification_settings: Optional[Dict] = Field(None)
    health_goals: Optional[Dict] = Field(None)
    
    class Config:
        """Pydantic configuration."""
        
        schema_extra = {
            "example": {
                "display_name": "John Smith",
                "bio": "Updated bio information",
                "timezone": "America/Los_Angeles",
                "health_goals": {
                    "target_sleep_hours": 7.5,
                    "target_steps": 12000
                }
            }
        }


class UserSession(BaseModel):
    """Model for user session information."""
    
    uid: str = Field(..., description="Firebase user ID")
    session_id: str = Field(..., description="Session identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None)
    device_info: Optional[Dict] = Field(None, description="Device information")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    
    class Config:
        """Pydantic configuration."""
        
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }