"""Authentication endpoints - AWS Cognito version."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field

from clarity.auth.dependencies import get_auth_provider, get_current_user
from clarity.core.exceptions import ProblemDetail
from clarity.models.auth import TokenResponse, UserLoginRequest
from clarity.ports.auth_ports import IAuthProvider
from clarity.services.cognito_auth_service import (
    CognitoAuthenticationService,
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
    auth_provider: IAuthProvider = Depends(get_auth_provider),
) -> TokenResponse:
    """Register a new user."""
    try:
        # Create authentication service
        auth_service = CognitoAuthenticationService(
            auth_provider, None
        )  # No document store needed for AWS

        # Register user - Note: May need to adapt this based on actual service interface
        return await auth_service.register_user(
            email=user_data.email,
            password=user_data.password,
            display_name=user_data.display_name,
        )

    except UserAlreadyExistsError as e:
        raise HTTPException(
            status_code=409,
            detail=ProblemDetail(
                type="user_already_exists",
                title="User Already Exists",
                detail=str(e),
                status=409,
                instance=f"https://api.clarity.health/requests/{id(e)}",
            ).model_dump(),
        ) from e
    except Exception as e:
        logger.exception("Registration failed")
        raise HTTPException(
            status_code=500,
            detail=ProblemDetail(
                type="registration_error",
                title="Registration Failed",
                detail="Failed to register user",
                status=500,
                instance=f"https://api.clarity.health/requests/{id(e)}",
            ).model_dump(),
        ) from e


@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: UserLoginRequest,
    auth_provider: IAuthProvider = Depends(get_auth_provider),
) -> TokenResponse:
    """Authenticate user and return access token."""
    try:
        # Create authentication service
        auth_service = CognitoAuthenticationService(
            auth_provider, None
        )  # No document store needed for AWS

        # Authenticate user - Note: May need to adapt this based on actual service interface
        return await auth_service.authenticate_user(
            email=credentials.email, password=credentials.password
        )

    except InvalidCredentialsError as e:
        raise HTTPException(
            status_code=401,
            detail=ProblemDetail(
                type="invalid_credentials",
                title="Invalid Credentials",
                detail=str(e),
                status=401,
                instance=f"https://api.clarity.health/requests/{id(e)}",
            ).model_dump(),
        )
    except EmailNotVerifiedError as e:
        raise HTTPException(
            status_code=403,
            detail=ProblemDetail(
                type="email_not_verified",
                title="Email Not Verified",
                detail=str(e),
                status=403,
                instance=f"https://api.clarity.health/requests/{id(e)}",
            ).model_dump(),
        )
    except Exception as e:
        logger.exception("Login failed")
        raise HTTPException(
            status_code=500,
            detail=ProblemDetail(
                type="authentication_error",
                title="Authentication Failed",
                detail="Failed to authenticate user",
                status=500,
                instance=f"https://api.clarity.health/requests/{id(e)}",
            ).model_dump(),
        ) from e


@router.get("/me")
async def get_current_user_info(
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
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
    current_user: dict[str, Any] = Depends(get_current_user),
    auth_provider: IAuthProvider = Depends(get_auth_provider),
) -> dict[str, Any]:
    """Update current user information."""
    try:
        # Create authentication service
        auth_service = CognitoAuthenticationService(auth_provider, None)

        # Get user ID
        user_id = current_user.get("uid", current_user.get("user_id"))

        # Update user - Note: May need to adapt this based on actual service interface
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
            detail=ProblemDetail(
                type="user_not_found",
                title="User Not Found",
                detail=str(e),
                status=404,
                instance=f"https://api.clarity.health/requests/{id(e)}",
            ).model_dump(),
        )
    except Exception as e:
        logger.exception("User update failed")
        raise HTTPException(
            status_code=500,
            detail=ProblemDetail(
                type="update_error",
                title="Update Failed",
                detail="Failed to update user",
                status=500,
                instance=f"https://api.clarity.health/requests/{id(e)}",
            ).model_dump(),
        ) from e


@router.post("/logout")
async def logout(
    request: Request,
    auth_provider: IAuthProvider = Depends(get_auth_provider),
) -> dict[str, str]:
    """Logout user (invalidate token if supported)."""
    try:
        # Check request format first - get body and auth header
        auth_header = request.headers.get("Authorization", "")

        # Check if request body is empty
        try:
            body = await request.json()
        except Exception:
            body = {}

        # If both body is empty and no auth header, this is a validation error
        if not body and not auth_header:
            raise HTTPException(
                status_code=422,
                detail=ProblemDetail(
                    type="validation_error",
                    title="Validation Error",
                    detail="Request body or Authorization header required for logout",
                    status=422,
                    instance=f"https://api.clarity.health/requests/{id(request)}",
                ).model_dump(),
            )

        # Now try to authenticate - if we have auth header
        current_user = None
        if auth_header:
            try:
                from clarity.auth.dependencies import get_current_user
                current_user = await get_current_user(request)
            except Exception:
                # Auth failed but we have a request, so it's an auth error not validation
                raise HTTPException(
                    status_code=401,
                    detail=ProblemDetail(
                        type="authentication_required",
                        title="Authentication Required",
                        detail="Invalid authentication credentials",
                        status=401,
                        instance=f"https://api.clarity.health/requests/{id(request)}",
                    ).model_dump(),
                )

        # Create authentication service
        auth_service = CognitoAuthenticationService(auth_provider, None)

        # Get token from header
        token = auth_header.replace("Bearer ", "") if auth_header else None

        # Logout user if we have a token
        if token:
            await auth_service.logout_user(token)

    except HTTPException:
        # Re-raise HTTP exceptions (validation errors, auth errors)
        raise
    except Exception as e:
        logger.exception("Logout failed")
        # Return success anyway - client should discard token
        return {"message": "Logout processed"}
    else:
        return {"message": "Successfully logged out"}


@router.get("/health")
async def auth_health() -> dict[str, str]:
    """Auth service health check."""
    try:
        # Simple health check - could be enhanced to check auth provider connectivity
        return {
            "status": "healthy",
            "service": "authentication",
            "version": "1.0.0"
        }
    except Exception:
        return {
            "status": "unhealthy",
            "service": "authentication",
            "version": "1.0.0"
        }


@router.post("/refresh")
async def refresh_token(
    request: Request,
    auth_provider: IAuthProvider = Depends(get_auth_provider),
) -> TokenResponse:
    """Refresh access token using refresh token."""
    try:
        # Create authentication service
        auth_service = CognitoAuthenticationService(auth_provider, None)

        # Get refresh token from request body or header
        auth_header = request.headers.get("Authorization", "")
        refresh_token = auth_header.replace("Bearer ", "") if auth_header else None

        if not refresh_token:
            # Try to get from request body
            body = await request.json()
            refresh_token = body.get("refresh_token")

        if not refresh_token:
            raise HTTPException(
                status_code=422,
                detail=ProblemDetail(
                    type="missing_refresh_token",
                    title="Missing Refresh Token",
                    detail="Refresh token is required",
                    status=422,
                    instance=f"https://api.clarity.health/requests/{id(request)}",
                ).model_dump(),
            )

        # Refresh the token
        return await auth_service.refresh_token(refresh_token)

    except Exception as e:
        logger.exception("Token refresh failed")
        raise HTTPException(
            status_code=500,
            detail=ProblemDetail(
                type="refresh_error",
                title="Token Refresh Failed",
                detail="Failed to refresh access token",
                status=500,
                instance=f"https://api.clarity.health/requests/{id(e)}",
            ).model_dump(),
        ) from e
