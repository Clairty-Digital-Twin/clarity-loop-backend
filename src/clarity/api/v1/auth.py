"""CLARITY Digital Twin Platform - Authentication API Endpoints.

RESTful API endpoints for user authentication including:
- User registration and login
- Token refresh and logout
- Email verification
- Session management
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer
from pydantic import ValidationError

from clarity.auth.firebase_auth import get_current_user
from clarity.auth.models import UserContext
from clarity.core.interfaces import IAuthProvider, IHealthDataRepository
from clarity.models.auth import (
    AuthErrorDetail,
    LoginResponse,
    RefreshTokenRequest,
    RegistrationResponse,
    TokenResponse,
    UserLoginRequest,
    UserRegistrationRequest,
    UserSessionResponse,
)
from clarity.services.auth_service import (
    AccountDisabledError,
    AuthenticationError,
    AuthenticationService,
    EmailNotVerifiedError,
    InvalidCredentialsError,
    UserAlreadyExistsError,
    UserNotFoundError,
)
from clarity.storage.firestore_client import FirestoreClient

# Configure logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Security scheme
security = HTTPBearer()

# Module-level singletons for dependencies (avoiding B008)
_auth_provider: IAuthProvider | None = None
_repository: IHealthDataRepository | None = None
_auth_service: AuthenticationService | None = None


def set_dependencies(
    auth_provider: IAuthProvider,
    repository: IHealthDataRepository,
    firestore_client: FirestoreClient | None = None,
) -> None:
    """Set dependencies for authentication endpoints.

    Args:
        auth_provider: Authentication provider
        repository: Health data repository
        firestore_client: Firestore client for user data
    """
    global _auth_provider, _repository, _auth_service  # noqa: PLW0603
    _auth_provider = auth_provider
    _repository = repository

    # Create authentication service
    if firestore_client:
        _auth_service = AuthenticationService(
            auth_provider=auth_provider,
            firestore_client=firestore_client,
        )
    else:
        # Fallback: create FirestoreClient from repository if available
        from clarity.storage.firestore_client import (  # noqa: PLC0415
            FirestoreHealthDataRepository,
        )

        if isinstance(repository, FirestoreHealthDataRepository):
            # Extract FirestoreClient from repository
            firestore_client = repository.client  # type: ignore[attr-defined]
            _auth_service = AuthenticationService(
                auth_provider=auth_provider,
                firestore_client=firestore_client,
            )
        else:
            logger.warning(
                "Authentication service not available - using mock implementation"
            )


def get_auth_service() -> AuthenticationService:
    """Get authentication service dependency."""
    if _auth_service is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service not configured",
        )
    return _auth_service


def get_device_info(request: Request) -> dict[str, Any]:
    """Extract device information from request headers."""
    return {
        "user_agent": request.headers.get("user-agent"),
        "ip_address": request.client.host if request.client else None,
        "device_type": "unknown",  # Could parse from user agent
        "os": "unknown",  # Could parse from user agent
        "browser": "unknown",  # Could parse from user agent
    }


# Authentication Endpoints


@router.post(
    "/register",
    response_model=RegistrationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with email and password",
    responses={
        201: {"description": "User registered successfully"},
        400: {"description": "Validation error or user already exists"},
        500: {"description": "Internal server error"},
    },
)
async def register_user(
    request_data: UserRegistrationRequest,
    request: Request,
) -> RegistrationResponse:
    """Register a new user account.

    Creates a new user with Firebase Authentication and stores additional
    user metadata in Firestore. Sends email verification if configured.
    """
    auth_service = get_auth_service()

    try:
        device_info = get_device_info(request)

        result = await auth_service.register_user(
            request=request_data,
            device_info=device_info,
        )

        logger.info("User registered: %s", result.email)
        return result

    except UserAlreadyExistsError as e:
        logger.warning("Registration attempt for existing user: %s", request_data.email)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "user_already_exists",
                "error_description": str(e),
                "error_details": [
                    AuthErrorDetail(
                        code="duplicate_email",
                        message="An account with this email already exists",
                        field="email",
                    ).model_dump()
                ],
            },
        ) from e

    except ValidationError as e:
        logger.warning("Registration validation error: %s", e)

        error_details = [
            AuthErrorDetail(
                code="validation_error",
                message=error["msg"],
                field=".".join(str(loc) for loc in error["loc"]),
            ).model_dump()
            for error in e.errors()
        ]

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "validation_error",
                "error_description": "Request validation failed",
                "error_details": error_details,
            },
        ) from e

    except AuthenticationError as e:
        logger.exception("Registration failed for %s", request_data.email)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "registration_failed",
                "error_description": "User registration failed",
            },
        ) from e


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="User login",
    description="Authenticate user with email and password",
    responses={
        200: {"description": "Login successful"},
        401: {"description": "Invalid credentials"},
        403: {"description": "Account disabled or email not verified"},
        404: {"description": "User not found"},
        500: {"description": "Internal server error"},
    },
)
async def login_user(
    request_data: UserLoginRequest,
    request: Request,
) -> LoginResponse:
    """Authenticate user and create session.

    Validates credentials with Firebase Authentication and returns
    access tokens and user session information.
    """
    auth_service = get_auth_service()

    try:
        device_info = get_device_info(request)
        ip_address = request.client.host if request.client else None

        result = await auth_service.login_user(
            request=request_data,
            device_info=device_info,
            ip_address=ip_address,
        )

        logger.info("User logged in: %s", request_data.email)
        return result

    except UserNotFoundError as e:
        logger.warning("Login attempt for non-existent user: %s", request_data.email)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "user_not_found",
                "error_description": "User account not found",
            },
        ) from e

    except InvalidCredentialsError as e:
        logger.warning("Invalid credentials for user: %s", request_data.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "invalid_credentials",
                "error_description": "Invalid email or password",
            },
        ) from e

    except EmailNotVerifiedError as e:
        logger.warning("Login attempt with unverified email: %s", request_data.email)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "email_not_verified",
                "error_description": "Email verification required before login",
            },
        ) from e

    except AccountDisabledError as e:
        logger.warning("Login attempt for disabled account: %s", request_data.email)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "account_disabled",
                "error_description": "Account has been disabled",
            },
        ) from e

    except AuthenticationError as e:
        logger.exception("Login failed for %s", request_data.email)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "login_failed",
                "error_description": "Authentication failed",
            },
        ) from e


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
    description="Get a new access token using refresh token",
    responses={
        200: {"description": "Token refreshed successfully"},
        401: {"description": "Invalid or expired refresh token"},
        500: {"description": "Internal server error"},
    },
)
async def refresh_token(
    request_data: RefreshTokenRequest,
) -> TokenResponse:
    """Refresh access token using refresh token.

    Validates the refresh token and returns a new access token.
    Implements token rotation for enhanced security.
    """
    auth_service = get_auth_service()

    try:
        result = await auth_service.refresh_access_token(request_data.refresh_token)

        logger.info("Access token refreshed successfully")
        return result

    except InvalidCredentialsError as e:
        logger.warning("Invalid refresh token: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "invalid_refresh_token",
                "error_description": "Invalid or expired refresh token",
            },
        ) from e

    except AuthenticationError as e:
        logger.exception("Token refresh failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "refresh_failed",
                "error_description": "Token refresh failed",
            },
        ) from e


@router.post(
    "/logout",
    summary="User logout",
    description="Logout user and revoke tokens",
    responses={
        200: {"description": "Logout successful"},
        401: {"description": "Invalid refresh token"},
        500: {"description": "Internal server error"},
    },
)
async def logout_user(
    request_data: RefreshTokenRequest,
) -> dict[str, str]:
    """Logout user and revoke session.

    Revokes the refresh token and ends the user session.
    All associated access tokens become invalid.
    """
    auth_service = get_auth_service()

    try:
        success = await auth_service.logout_user(request_data.refresh_token)

        if success:
            logger.info("User logged out successfully")
            return {"message": "Logout successful"}

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "logout_failed",
                "error_description": "Logout failed - invalid refresh token",
            },
        )

    except AuthenticationError as e:
        logger.exception("Logout failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "logout_failed",
                "error_description": "Logout operation failed",
            },
        ) from e


@router.get(
    "/me",
    response_model=UserSessionResponse,
    summary="Get current user info",
    description="Get information about the currently authenticated user",
    responses={
        200: {"description": "User information retrieved"},
        401: {"description": "Authentication required"},
        404: {"description": "User not found"},
        500: {"description": "Internal server error"},
    },
)
async def get_current_user_info(
    current_user: UserContext = Depends(get_current_user),
) -> UserSessionResponse:
    """Get current user information.

    Returns detailed information about the currently authenticated user
    based on the JWT token in the Authorization header.
    """
    auth_service = get_auth_service()

    try:
        user_info = await auth_service.get_user_by_id(current_user.user_id)

        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "user_not_found",
                    "error_description": "User information not found",
                },
            )

        return user_info

    except AuthenticationError as e:
        logger.exception("Failed to get user info for %s", current_user.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "user_info_failed",
                "error_description": "Failed to retrieve user information",
            },
        ) from e


@router.post(
    "/verify-email",
    summary="Verify email address",
    description="Verify user email with verification code",
    responses={
        200: {"description": "Email verified successfully"},
        400: {"description": "Invalid verification code"},
        500: {"description": "Internal server error"},
    },
)
async def verify_email(
    verification_code: str,
) -> dict[str, str]:
    """Verify user email address.

    Validates the email verification code sent to the user's email.
    Updates the user's email verification status upon success.
    """
    auth_service = get_auth_service()

    try:
        success = await auth_service.verify_email(verification_code)

        if success:
            logger.info("Email verification successful")
            return {"message": "Email verified successfully"}

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "verification_failed",
                "error_description": "Invalid or expired verification code",
            },
        )

    except AuthenticationError as e:
        logger.exception("Email verification failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "verification_failed",
                "error_description": "Email verification failed",
            },
        ) from e


# Health check endpoint for authentication service
@router.get(
    "/health",
    summary="Authentication service health check",
    description="Check if authentication service is healthy",
    include_in_schema=False,
)
async def auth_health_check() -> dict[str, Any]:
    """Health check for authentication service."""
    try:
        # Test if auth service is available
        auth_service = get_auth_service()

        return {
            "status": "healthy",
            "service": "authentication",
            "dependencies": {
                "auth_provider": _auth_provider is not None,
                "repository": _repository is not None,
                "auth_service": auth_service is not None,
            },
        }

    except Exception as e:
        logger.exception("Authentication health check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "service": "authentication",
                "error": str(e),
            },
        ) from e
