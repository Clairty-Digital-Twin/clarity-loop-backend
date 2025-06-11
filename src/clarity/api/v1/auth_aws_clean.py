"""Authentication endpoints - AWS Cognito version without Firebase dependencies."""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field

from clarity.auth.dependencies import get_auth_provider, get_current_user  
from clarity.core.exceptions import ProblemDetails
from clarity.models.auth import TokenResponse, UserLogin
from clarity.ports.auth_ports import AuthenticationProvider
from clarity.services.auth_service import (
    AuthenticationService,
    EmailNotVerifiedError,
    InvalidCredentialsError,
    UserAlreadyExistsError,
    UserNotFoundError,
)

# Configure logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


class UserRegister(BaseModel):
    """User registration request model."""
    
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    display_name: str | None = Field(None, description="Optional display name")


class UserUpdate(BaseModel):
    """User update request model."""
    
    display_name: str | None = Field(None, description="Display name to update")
    email: str | None = Field(None, description="New email address")


@router.post("/register", response_model=TokenResponse)
async def register(
    user_data: UserRegister,
    auth_provider: AuthenticationProvider = Depends(get_auth_provider),
) -> TokenResponse:
    """Register a new user."""
    try:
        # Create authentication service
        auth_service = AuthenticationService(auth_provider, None)  # No Firestore needed for AWS
        
        # Register user
        token_response = await auth_service.register_user(
            email=user_data.email,
            password=user_data.password,
            display_name=user_data.display_name,
        )
        
        return token_response
        
    except UserAlreadyExistsError as e:
        raise HTTPException(
            status_code=409,
            detail=ProblemDetails(
                type="user_already_exists",
                title="User Already Exists",
                detail=str(e),
                status=409,
            ).model_dump(),
        )
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=ProblemDetails(
                type="registration_error",
                title="Registration Failed",
                detail="Failed to register user",
                status=500,
            ).model_dump(),
        )


@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: UserLogin,
    auth_provider: AuthenticationProvider = Depends(get_auth_provider),
) -> TokenResponse:
    """Authenticate user and return access token."""
    try:
        # Create authentication service
        auth_service = AuthenticationService(auth_provider, None)  # No Firestore needed for AWS
        
        # Authenticate user
        token_response = await auth_service.authenticate_user(
            email=credentials.email, password=credentials.password
        )
        
        return token_response
        
    except InvalidCredentialsError as e:
        raise HTTPException(
            status_code=401,
            detail=ProblemDetails(
                type="invalid_credentials",
                title="Invalid Credentials",
                detail=str(e),
                status=401,
            ).model_dump(),
        )
    except EmailNotVerifiedError as e:
        raise HTTPException(
            status_code=403,
            detail=ProblemDetails(
                type="email_not_verified",
                title="Email Not Verified",
                detail=str(e),
                status=403,
            ).model_dump(),
        )
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=ProblemDetails(
                type="authentication_error",
                title="Authentication Failed",
                detail="Failed to authenticate user",
                status=500,
            ).model_dump(),
        )


@router.get("/me")
async def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get current user information."""
    return {
        "user_id": current_user.get("uid", current_user.get("user_id")),
        "email": current_user.get("email"),
        "email_verified": current_user.get("email_verified", True),
        "display_name": current_user.get("display_name"),
        "auth_provider": current_user.get("auth_provider", "cognito"),
    }


@router.put("/me")
async def update_user(
    updates: UserUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_provider: AuthenticationProvider = Depends(get_auth_provider),
) -> Dict[str, Any]:
    """Update current user information."""
    try:
        # Create authentication service
        auth_service = AuthenticationService(auth_provider, None)
        
        # Get user ID
        user_id = current_user.get("uid", current_user.get("user_id"))
        
        # Update user
        updated_user = await auth_service.update_user(
            user_id=user_id,
            display_name=updates.display_name,
            email=updates.email,
        )
        
        return {
            "user_id": updated_user.get("uid", updated_user.get("user_id")),
            "email": updated_user.get("email"),
            "display_name": updated_user.get("display_name"),
            "updated": True,
        }
        
    except UserNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=ProblemDetails(
                type="user_not_found",
                title="User Not Found",
                detail=str(e),
                status=404,
            ).model_dump(),
        )
    except Exception as e:
        logger.error(f"User update failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=ProblemDetails(
                type="update_error",
                title="Update Failed",
                detail="Failed to update user",
                status=500,
            ).model_dump(),
        )


@router.post("/logout")
async def logout(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_provider: AuthenticationProvider = Depends(get_auth_provider),
) -> Dict[str, str]:
    """Logout user (invalidate token if supported)."""
    try:
        # Create authentication service
        auth_service = AuthenticationService(auth_provider, None)
        
        # Get token from header
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.replace("Bearer ", "") if auth_header else None
        
        # Logout user
        await auth_service.logout_user(token)
        
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        # Return success anyway - client should discard token
        return {"message": "Logout processed"}