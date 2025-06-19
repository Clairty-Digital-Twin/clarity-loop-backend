"""Authentication endpoints - AWS Cognito version."""

import json
import logging
import os
from typing import Any

import boto3
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from clarity.auth.aws_cognito_provider import CognitoAuthProvider
from clarity.auth.dependencies import get_auth_provider, get_current_user
from clarity.auth.dependencies import get_current_user as get_user_func
from clarity.auth.lockout_service import (
    AccountLockoutError,
    AccountLockoutService,
    get_lockout_service,
)
from clarity.core.constants import (
    AUTH_HEADER_TYPE_BEARER,
    AUTH_SCOPE_FULL_ACCESS,
    AUTH_TOKEN_DEFAULT_EXPIRY_SECONDS,
)
from clarity.core.exceptions import (
    AuthenticationError as CoreAuthError,
)
from clarity.core.exceptions import (
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

# Lockout service will be injected as a dependency

# Initialize CloudWatch client
cloudwatch = boto3.client(
    "cloudwatch", region_name=os.getenv("AWS_REGION", "us-east-1")
)
USER_POOL_ID = os.getenv("COGNITO_USER_POOL_ID", "")

# Create rate limiter for auth endpoints
auth_limiter = Limiter(key_func=get_remote_address)


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
@auth_limiter.limit("5/hour")  # Very strict limit for registration
async def register(
    request: Request,
    user_data: UserRegister,
    auth_provider: IAuthProvider = Depends(get_auth_provider),
    _lockout_service: AccountLockoutService = Depends(get_lockout_service),
) -> TokenResponse | JSONResponse:
    """Register a new user."""
    _ = request  # Used by rate limiter
    # Check if self-signup is enabled
    enable_self_signup = os.getenv("ENABLE_SELF_SIGNUP", "false").lower() == "true"
    if not enable_self_signup:
        raise HTTPException(
            status_code=403,
            detail=ProblemDetail(
                type="self_signup_disabled",
                title="Self Sign-up Disabled",
                detail=(
                    "Self-registration is currently disabled. "
                    "Please contact an administrator to create an account."
                ),
                status=403,
                instance="https://api.clarity.health/auth/register",
            ).model_dump(),
        )

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
    except InvalidCredentialsError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e
    except EmailNotVerifiedError:
        # Return 202 Accepted when email verification is required
        return JSONResponse(
            status_code=202, content={"requires_email_verification": True}
        )
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
@auth_limiter.limit("10/minute")  # Stricter rate limit for login endpoint
async def login(
    request: Request,
    credentials: UserLoginRequest,
    auth_provider: IAuthProvider = Depends(get_auth_provider),
    lockout_service: AccountLockoutService = Depends(get_lockout_service),
) -> TokenResponse:
    """Authenticate user and return access token."""
    # Get client IP for lockout tracking
    client_ip = request.client.host if request.client else "unknown"

    # Debug logging for request body
    try:
        body_bytes = await request.body()
        logger.warning("🔍 LOGIN REQUEST DEBUG:")
        logger.warning("  Raw body bytes: %r", body_bytes)
        logger.warning("  Body length: %d bytes", len(body_bytes))
        logger.warning("  Body as string: %s", body_bytes.decode("utf-8"))
        logger.warning("  Parsed credentials: email=%s", credentials.email)
        logger.warning("  Client IP: %s", client_ip)
    except Exception as e:
        logger.exception("Failed to log request body: %s", e)

    # Check for account lockout BEFORE attempting authentication
    try:
        await lockout_service.check_lockout(credentials.email)
        logger.debug("✅ Account lockout check passed for %s", credentials.email)
    except AccountLockoutError as e:
        logger.warning("🔒 Account locked: %s", e)
        raise HTTPException(
            status_code=429,
            detail=ProblemDetail(
                type="account_locked",
                title="Account Temporarily Locked",
                detail=str(e),
                status=429,
                instance=f"https://api.clarity.health/requests/{id(e)}",
            ).model_dump(),
        ) from e

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

        # Authentication successful - reset failed attempts
        await lockout_service.reset_attempts(credentials.email)
        logger.info(
            "✅ Login successful for %s, lockout attempts reset", credentials.email
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
        logger.warning(
            "Authentication failed for user: %s. Returning 401.", credentials.email
        )

        # Track failed attempt for lockout protection
        try:
            await lockout_service.record_failed_attempt(credentials.email, client_ip)
            logger.info(
                "🚨 Failed login attempt recorded for %s from %s",
                credentials.email,
                client_ip,
            )

            # Check if account just got locked and emit CloudWatch metric
            if await lockout_service.is_locked(credentials.email):
                try:
                    cloudwatch.put_metric_data(
                        Namespace="Clarity/Auth",
                        MetricData=[
                            {
                                "MetricName": "AccountLockout",
                                "Dimensions": [
                                    {"Name": "UserPoolId", "Value": USER_POOL_ID}
                                ],
                                "Value": 1,
                                "Unit": "Count",
                            }
                        ],
                    )
                    logger.warning(
                        "📊 CloudWatch metric emitted: Account lockout for %s",
                        credentials.email,
                    )
                except Exception as metric_error:
                    logger.exception(
                        "Failed to emit CloudWatch metric: %s", metric_error
                    )

        except Exception as lockout_error:
            # Don't let lockout service errors block the auth response
            logger.exception("Failed to record lockout attempt: %s", lockout_error)

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
    _auth_provider: IAuthProvider = Depends(get_auth_provider),
) -> LogoutResponse:
    """Logout user (invalidate token if supported)."""
    # Check request format first - get body and auth header
    auth_header = request.headers.get("Authorization", "")

    # Check if request body is empty
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError, TypeError):
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
                # Auth failed but we have a request,
                # so it's an auth error not validation
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
    except Exception:  # noqa: BLE001 - Health check should catch all exceptions
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
        except (json.JSONDecodeError, ValueError, TypeError):
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
                    # Cognito doesn't rotate refresh tokens
                    refresh_token=refresh_token_str,
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


class EmailConfirmationRequest(BaseModel):
    """Email confirmation request model."""

    email: EmailStr = Field(..., description="User email address")
    code: str = Field(..., description="6-digit confirmation code from email")


class ResendConfirmationRequest(BaseModel):
    """Resend confirmation code request model."""

    email: EmailStr = Field(..., description="User email address")


class ForgotPasswordRequest(BaseModel):
    """Forgot password request model."""

    email: EmailStr = Field(..., description="User email address")


class ResetPasswordRequest(BaseModel):
    """Reset password request model."""

    email: EmailStr = Field(..., description="User email address")
    code: str = Field(..., description="6-digit reset code from email")
    new_password: str = Field(..., min_length=8, description="New password")


class StatusResponse(BaseModel):
    """Generic status response model."""

    status: str = Field(..., description="Operation status")


@router.post("/confirm-email", response_model=StatusResponse)
async def confirm_email(
    request: EmailConfirmationRequest,
    auth_provider: IAuthProvider = Depends(get_auth_provider),
) -> StatusResponse:
    """Confirm user email with verification code from Cognito."""
    if not isinstance(auth_provider, CognitoAuthProvider):
        raise HTTPException(
            status_code=500, detail="Invalid authentication provider configuration"
        )

    try:
        # Use Cognito client to confirm sign up
        auth_provider.cognito_client.confirm_sign_up(
            ClientId=auth_provider.client_id,
            Username=request.email,
            ConfirmationCode=request.code,
        )

        return StatusResponse(status="confirmed")

    except auth_provider.cognito_client.exceptions.CodeMismatchException as e:
        raise HTTPException(
            status_code=400,
            detail=ProblemDetail(
                type="invalid_code",
                title="Invalid Confirmation Code",
                detail="The confirmation code is incorrect or expired",
                status=400,
                instance=f"https://api.clarity.health/requests/{id(request)}",
            ).model_dump(),
        ) from e
    except auth_provider.cognito_client.exceptions.UserNotFoundException as e:
        raise HTTPException(
            status_code=404,
            detail=ProblemDetail(
                type="user_not_found",
                title="User Not Found",
                detail="No user found with this email address",
                status=404,
                instance=f"https://api.clarity.health/requests/{id(request)}",
            ).model_dump(),
        ) from e
    except Exception as e:
        logger.exception("Email confirmation failed")
        raise HTTPException(
            status_code=500,
            detail=ProblemDetail(
                type="confirmation_error",
                title="Confirmation Failed",
                detail="Failed to confirm email address",
                status=500,
                instance=f"https://api.clarity.health/requests/{id(e)}",
            ).model_dump(),
        ) from e


@router.post("/resend-confirmation", response_model=StatusResponse)
async def resend_confirmation(
    request: ResendConfirmationRequest,
    auth_provider: IAuthProvider = Depends(get_auth_provider),
) -> StatusResponse:
    """Resend email confirmation code."""
    if not isinstance(auth_provider, CognitoAuthProvider):
        raise HTTPException(
            status_code=500, detail="Invalid authentication provider configuration"
        )

    try:
        # Use Cognito client to resend confirmation code
        auth_provider.cognito_client.resend_confirmation_code(
            ClientId=auth_provider.client_id,
            Username=request.email,
        )

        return StatusResponse(status="sent")

    except auth_provider.cognito_client.exceptions.UserNotFoundException as e:
        raise HTTPException(
            status_code=404,
            detail=ProblemDetail(
                type="user_not_found",
                title="User Not Found",
                detail="No user found with this email address",
                status=404,
                instance=f"https://api.clarity.health/requests/{id(request)}",
            ).model_dump(),
        ) from e
    except Exception as e:
        logger.exception("Resend confirmation failed")
        raise HTTPException(
            status_code=500,
            detail=ProblemDetail(
                type="resend_error",
                title="Resend Failed",
                detail="Failed to resend confirmation code",
                status=500,
                instance=f"https://api.clarity.health/requests/{id(e)}",
            ).model_dump(),
        ) from e


@router.post("/forgot-password", response_model=StatusResponse)
async def forgot_password(
    request: ForgotPasswordRequest,
    auth_provider: IAuthProvider = Depends(get_auth_provider),
) -> StatusResponse:
    """Initiate password reset process."""
    if not isinstance(auth_provider, CognitoAuthProvider):
        raise HTTPException(
            status_code=500, detail="Invalid authentication provider configuration"
        )

    try:
        # Use Cognito client to initiate forgot password
        auth_provider.cognito_client.forgot_password(
            ClientId=auth_provider.client_id,
            Username=request.email,
        )

        return StatusResponse(status="sent")

    except auth_provider.cognito_client.exceptions.UserNotFoundException as e:
        raise HTTPException(
            status_code=404,
            detail=ProblemDetail(
                type="user_not_found",
                title="User Not Found",
                detail="No user found with this email address",
                status=404,
                instance=f"https://api.clarity.health/requests/{id(request)}",
            ).model_dump(),
        ) from e
    except Exception as e:
        logger.exception("Forgot password failed")
        raise HTTPException(
            status_code=500,
            detail=ProblemDetail(
                type="forgot_password_error",
                title="Password Reset Failed",
                detail="Failed to initiate password reset",
                status=500,
                instance=f"https://api.clarity.health/requests/{id(e)}",
            ).model_dump(),
        ) from e


@router.post("/reset-password", response_model=StatusResponse)
async def reset_password(
    request: ResetPasswordRequest,
    auth_provider: IAuthProvider = Depends(get_auth_provider),
) -> StatusResponse:
    """Reset password with confirmation code."""
    if not isinstance(auth_provider, CognitoAuthProvider):
        raise HTTPException(
            status_code=500, detail="Invalid authentication provider configuration"
        )

    try:
        # Use Cognito client to confirm forgot password
        auth_provider.cognito_client.confirm_forgot_password(
            ClientId=auth_provider.client_id,
            Username=request.email,
            ConfirmationCode=request.code,
            Password=request.new_password,
        )

        return StatusResponse(status="reset")

    except auth_provider.cognito_client.exceptions.CodeMismatchException as e:
        raise HTTPException(
            status_code=400,
            detail=ProblemDetail(
                type="invalid_code",
                title="Invalid Reset Code",
                detail="The reset code is incorrect or expired",
                status=400,
                instance=f"https://api.clarity.health/requests/{id(request)}",
            ).model_dump(),
        ) from e
    except auth_provider.cognito_client.exceptions.UserNotFoundException as e:
        raise HTTPException(
            status_code=404,
            detail=ProblemDetail(
                type="user_not_found",
                title="User Not Found",
                detail="No user found with this email address",
                status=404,
                instance=f"https://api.clarity.health/requests/{id(request)}",
            ).model_dump(),
        ) from e
    except Exception as e:
        logger.exception("Password reset failed")
        raise HTTPException(
            status_code=500,
            detail=ProblemDetail(
                type="reset_password_error",
                title="Password Reset Failed",
                detail="Failed to reset password",
                status=500,
                instance=f"https://api.clarity.health/requests/{id(e)}",
            ).model_dump(),
        ) from e
