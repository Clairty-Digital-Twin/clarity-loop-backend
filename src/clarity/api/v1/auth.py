"""Authentication endpoints - AWS Cognito version."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field

from clarity.auth.aws_cognito_provider import CognitoAuthProvider
from clarity.auth.dependencies import get_auth_provider, get_current_user
from clarity.auth.dependencies import get_current_user as get_user_func
from clarity.core.constants import (
    AUTH_HEADER_TYPE_BEARER,
    AUTH_SCOPE_FULL_ACCESS,
    AUTH_TOKEN_DEFAULT_EXPIRY_SECONDS,
)
from clarity.core.exceptions import (
    AuthenticationError as CoreAuthError,
    EmailNotVerifiedError,
    InvalidCredentialsError,
    ProblemDetail,
    UserAlreadyExistsError,
    UserNotFoundError,
)
from clarity.models.auth import TokenResponse, UserLoginRequest
from clarity.ports.auth_ports import IAuthProvider

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


class UserInfoResponse(BaseModel):
    """User information response model."""

    user_id: str = Field(..., description="Unique user identifier")
    email: str | None = Field(None, description="User email address")
    email_verified: bool = Field(..., description="Email verification status")
    display_name: str | None = Field(None, description="User display name")
    auth_provider: str = Field(..., description="Authentication provider")


class UserUpdateResponse(BaseModel):
    """User update response model."""

    user_id: str = Field(..., description="Unique user identifier")
    email: str | None = Field(None, description="User email address")
    display_name: str | None = Field(None, description="User display name")
    updated: bool = Field(..., description="Update success status")


class LogoutResponse(BaseModel):
    """Logout response model."""

    message: str = Field(..., description="Logout status message")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")


@router.post("/register", response_model=TokenResponse)
async def register(
    user_data: UserRegister,
    auth_provider: IAuthProvider = Depends(get_auth_provider),
) -> TokenResponse:
    """Register a new user."""
    # Validate auth provider before try block
    if not isinstance(auth_provider, CognitoAuthProvider):
        raise HTTPException(
            status_code=500, detail="Invalid authentication provider configuration"
        )

    try:
        # Create user in Cognito
        user = await auth_provider.create_user(
            email=user_data.email,
            password=user_data.password,
            display_name=user_data.display_name,
        )

        # Now authenticate to get tokens
        tokens = await auth_provider.authenticate(
            email=user_data.email,
            password=user_data.password,
        )

    except UserAlreadyExistsError as e:
        raise HTTPException(
            status_code=409,
            detail=str(e),
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

    # Validate results outside try block
    if not user:
        raise HTTPException(status_code=500, detail="Failed to create user")

    if not tokens:
        raise HTTPException(
            status_code=500, detail="Failed to authenticate after registration"
        )

    # Return token response
    return TokenResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        token_type=AUTH_HEADER_TYPE_BEARER,
        expires_in=tokens.get("expires_in", AUTH_TOKEN_DEFAULT_EXPIRY_SECONDS),
        scope=AUTH_SCOPE_FULL_ACCESS,
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    credentials: UserLoginRequest,
    auth_provider: IAuthProvider = Depends(get_auth_provider),
) -> TokenResponse:
    """Authenticate user and return access token."""
    # Debug logging for request body
    try:
        body_bytes = await request.body()
        logger.warning("🔍 LOGIN REQUEST DEBUG:")
        logger.warning(f"  Raw body bytes: {body_bytes!r}")
        logger.warning(f"  Body length: {len(body_bytes)} bytes")
        logger.warning(f"  Body as string: {body_bytes.decode('utf-8')}")
        logger.warning(f"  Parsed credentials: email={credentials.email}")
    except Exception as e:
        logger.exception(f"Failed to log request body: {e}")

    # Validate auth provider before try block
    if not isinstance(auth_provider, CognitoAuthProvider):
        raise HTTPException(
            status_code=500, detail="Invalid authentication provider configuration"
        )

    try:
        # Authenticate user
        tokens = await auth_provider.authenticate(
            email=credentials.email,
            password=credentials.password,
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
        ) from e
    except (InvalidCredentialsError, CoreAuthError) as e:
        # Both InvalidCredentialsError from the service layer and CoreAuthError
        # from the provider layer indicate a client-side authentication failure.
        # These should consistently result in a 401 Unauthorized response.
        # We use a generic error message to avoid leaking details about the failure.
        logger.warning("Authentication failed for user: %s. Returning 401.", credentials.email)
        raise HTTPException(
            status_code=401,
            detail=ProblemDetail(
                type="invalid_credentials",
                title="Invalid Credentials",
                detail="Invalid email or password.",
                status=401,
                instance=f"https://api.clarity.health/requests/{id(e)}",
            ).model_dump(),
        ) from e
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

    # Validate result outside try block
    if not tokens:
        raise HTTPException(status_code=500, detail="Failed to authenticate user")

    # Return token response
    return TokenResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        token_type=AUTH_HEADER_TYPE_BEARER,
        expires_in=tokens.get("expires_in", AUTH_TOKEN_DEFAULT_EXPIRY_SECONDS),
        scope=AUTH_SCOPE_FULL_ACCESS,
    )


@router.get("/me", response_model=UserInfoResponse)
async def get_current_user_info(
    current_user: dict[str, Any] = Depends(get_current_user),
) -> UserInfoResponse:
    """Get current user information."""
    return UserInfoResponse(
        user_id=current_user.get("uid", current_user.get("user_id", "")),
        email=current_user.get("email"),
        email_verified=current_user.get("email_verified", True),
        display_name=current_user.get("display_name"),
        auth_provider=current_user.get("auth_provider", "cognito"),
    )


@router.put("/me", response_model=UserUpdateResponse)
async def update_user(
    updates: UserUpdate,
    current_user: dict[str, Any] = Depends(get_current_user),
    auth_provider: IAuthProvider = Depends(get_auth_provider),
) -> UserUpdateResponse:
    """Update current user information."""
    # Validate auth provider before try block
    if not isinstance(auth_provider, CognitoAuthProvider):
        raise HTTPException(
            status_code=500, detail="Invalid authentication provider configuration"
        )

    # Get user ID and validate
    user_id = current_user.get("uid", current_user.get("user_id", ""))
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in token")

    try:

        # Build update kwargs
        update_kwargs: dict[str, Any] = {}
        if updates.display_name is not None:
            update_kwargs["display_name"] = updates.display_name
        if updates.email is not None:
            update_kwargs["email"] = updates.email

        # Update user
        updated_user = await auth_provider.update_user(uid=user_id, **update_kwargs)

    except UserNotFoundError:
        raise
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

    # Validate result outside try block
    if not updated_user:
        msg = f"User {user_id} not found"
        raise UserNotFoundError(msg)

    return UserUpdateResponse(
        user_id=updated_user.uid,
        email=updated_user.email,
        display_name=updated_user.display_name,
        updated=True,
    )


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    request: Request,
    auth_provider: IAuthProvider = Depends(get_auth_provider),
) -> LogoutResponse:
    """Logout user (invalidate token if supported)."""
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

    try:

        # Now try to authenticate - if we have auth header
        if auth_header:
            try:
                _ = get_user_func(request)
            except Exception as auth_err:
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
                ) from auth_err

        # For AWS Cognito, logout is typically handled client-side
        # by removing tokens. Server-side we can optionally revoke tokens
        # if using refresh tokens
        logger.info("Logout request processed")

    except HTTPException:
        # Re-raise HTTP exceptions (validation errors, auth errors)
        raise
    except Exception:
        logger.exception("Logout failed")
        # Return success anyway - client should discard token
        return LogoutResponse(message="Logout processed")
    else:
        return LogoutResponse(message="Successfully logged out")


@router.get("/health", response_model=HealthResponse)
async def auth_health() -> HealthResponse:
    """Auth service health check."""
    try:
        # Simple health check - could be enhanced to check auth provider connectivity
        return HealthResponse(
            status="healthy", service="authentication", version="1.0.0"
        )
    except Exception:
        return HealthResponse(
            status="unhealthy", service="authentication", version="1.0.0"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: Request,
    auth_provider: IAuthProvider = Depends(get_auth_provider),
) -> TokenResponse:
    """Refresh access token using refresh token."""
    # Get refresh token from request body or header
    auth_header = request.headers.get("Authorization", "")
    refresh_token_str = auth_header.replace("Bearer ", "") if auth_header else None

    if not refresh_token_str:
        # Try to get from request body
        try:
            body = await request.json()
            refresh_token_str = body.get("refresh_token")
        except Exception:
            refresh_token_str = None

    if not refresh_token_str:
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

    # For AWS Cognito, we need to use the boto3 client directly
    # Since refresh token handling is different
    if not isinstance(auth_provider, CognitoAuthProvider):
        raise HTTPException(
            status_code=500, detail="Invalid authentication provider configuration"
        )

    try:

        # Use Cognito's refresh token flow
        client = auth_provider.cognito_client

        try:
            response = client.initiate_auth(
                ClientId=auth_provider.client_id,
                AuthFlow="REFRESH_TOKEN_AUTH",
                AuthParameters={
                    "REFRESH_TOKEN": refresh_token_str,
                },
            )

            if "AuthenticationResult" in response:
                result = response["AuthenticationResult"]
                return TokenResponse(
                    access_token=result["AccessToken"],
                    refresh_token=refresh_token_str,  # Cognito doesn't rotate refresh tokens
                    token_type=AUTH_HEADER_TYPE_BEARER,
                    expires_in=result.get(
                        "ExpiresIn", AUTH_TOKEN_DEFAULT_EXPIRY_SECONDS
                    ),
                    scope=AUTH_SCOPE_FULL_ACCESS,
                )
            raise HTTPException(status_code=500, detail="Failed to refresh token")
        except client.exceptions.NotAuthorizedException as auth_err:
            raise HTTPException(
                status_code=401, detail="Invalid refresh token"
            ) from auth_err

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
