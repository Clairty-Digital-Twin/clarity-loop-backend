"""Authentication dependencies for Clean Architecture.

This module provides the single source of truth for authentication dependencies
following Robert C. Martin's principles and security best practices.
Uses AWS Cognito for authentication backend.
"""

# removed - breaks FastAPI

import logging
from typing import Annotated, Any, cast

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer

from clarity.auth.aws_cognito_provider import get_cognito_provider
from clarity.auth.lockout_service import get_lockout_service  # noqa: F401
from clarity.auth.modal_auth_fix import get_user_context
from clarity.models.auth import UserContext
from clarity.models.user import User

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)


def get_authenticated_user(
    request: Request,
) -> UserContext:
    """Get authenticated user with full context from middleware.

    This is the primary authentication dependency that should be used
    for all protected endpoints. It returns a UserContext which includes:
    - User ID from AWS Cognito
    - Email and verification status
    - Role and permissions from DynamoDB
    - Additional user metadata

    The middleware handles:
    - JWT token verification via AWS Cognito
    - DynamoDB record creation (if needed)
    - User context enrichment

    Args:
        request: FastAPI request object

    Returns:
        UserContext with complete user information

    Raises:
        HTTPException: 401 if not authenticated
    """
    # Check contextvars first (Modal doesn't propagate request.state properly)
    user_context = get_user_context()

    if user_context:
        logger.info("✅ User authenticated via contextvars: %s", user_context.user_id)
        return user_context

    # Fallback to checking request.state (for local development)
    if not hasattr(request.state, "user") or request.state.user is None:
        logger.warning(
            "No user context in contextvars or request.state for path: %s",
            request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_context = request.state.user

    if not isinstance(user_context, UserContext):
        logger.error("Invalid user context type: %s", type(user_context))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid authentication state",
        )

    return user_context


def get_current_user(request: Request) -> dict[str, Any]:
    """Get current user for backward compatibility.

    This is an alias for get_authenticated_user but returns a dict format
    for compatibility with legacy code.

    Args:
        request: FastAPI request object

    Returns:
        User information as dict

    Raises:
        HTTPException: 401 if not authenticated
    """
    user_context = get_authenticated_user(request)

    # Convert UserContext to dict format for backward compatibility
    return {
        "uid": user_context.user_id,
        "user_id": user_context.user_id,
        "email": user_context.email,
        "email_verified": user_context.is_verified,
        "display_name": getattr(user_context, "display_name", None),
        "auth_provider": "cognito",
        "role": getattr(user_context, "role", "user"),
        "is_active": getattr(user_context, "is_active", True),
        "created_at": user_context.created_at,
        "last_login": user_context.last_login,
    }


def get_auth_provider() -> Any:
    """Get authentication provider for dependency injection.

    This creates a simple auth provider for AWS Cognito.
    For more complex setups, this would come from a DI container.

    Returns:
        Authentication provider instance
    """
    return get_cognito_provider()


def get_optional_user(
    request: Request,
) -> UserContext | None:
    """Get authenticated user if available, None otherwise.

    Use this for endpoints that have optional authentication.

    Args:
        request: FastAPI request object
        credentials: Optional bearer token

    Returns:
        UserContext if authenticated, None otherwise
    """
    if not hasattr(request.state, "user") or request.state.user is None:
        return None

    user_context = request.state.user
    if not isinstance(user_context, UserContext):
        return None

    return user_context


# Type aliases for cleaner function signatures
AuthenticatedUser = Annotated[UserContext, Depends(get_authenticated_user)]
OptionalUser = Annotated[UserContext | None, Depends(get_optional_user)]


# Specialized dependencies for specific requirements
def require_verified_email(
    user: UserContext = Depends(get_authenticated_user),
) -> UserContext:
    """Require authenticated user with verified email.

    Args:
        user: Authenticated user context

    Returns:
        UserContext if email is verified

    Raises:
        HTTPException: 403 if email not verified
    """
    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required",
        )
    return user


def require_active_account(
    user: UserContext = Depends(get_authenticated_user),
) -> UserContext:
    """Require authenticated user with active account.

    Args:
        user: Authenticated user context

    Returns:
        UserContext if account is active

    Raises:
        HTTPException: 403 if account is not active
    """
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is suspended or inactive",
        )
    return user


# Convenience function for transitioning from old User model
def user_context_to_simple_user(context: UserContext) -> User:
    """Convert UserContext to simple User model for backward compatibility.

    This is a temporary helper for transitioning endpoints.

    Args:
        context: Full user context

    Returns:
        Simple User model
    """
    return User(
        uid=context.user_id,
        email=context.email,
        display_name=getattr(context, "display_name", None),
        cognito_token="",  # Not available from context  # nosec B106
        cognito_token_exp=None,
        role=getattr(context, "role", "user"),
        created_at=context.created_at,
        last_login=context.last_login,
        is_active=getattr(context, "is_active", True),
        metadata=getattr(context, "metadata", {}),
    )


def get_websocket_user(token: str, request: Request) -> UserContext:
    """Get authenticated user for WebSocket connections.

    WebSocket connections pass the token as a query parameter rather than
    in headers, so we need special handling.

    Args:
        token: AWS Cognito JWT token from query parameter
        request: FastAPI request object (for state access)

    Returns:
        UserContext with complete user information

    Raises:
        HTTPException: 401 if token is invalid
    """
    # WebSocket connections need special handling for auth
    # Modify the ASGI scope headers directly to add authorization
    raw_headers = list(request.scope["headers"])
    raw_headers.append((b"authorization", f"Bearer {token}".encode()))
    request.scope["headers"] = raw_headers

    # The middleware will process this and set request.state.user
    # We can then retrieve it
    if not hasattr(request.state, "user") or request.state.user is None:
        logger.warning("WebSocket authentication failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )

    return cast("UserContext", request.state.user)
