"""Authentication dependencies for Clean Architecture.

This module provides the single source of truth for authentication dependencies
following Robert C. Martin's principles and security best practices.
"""

import logging
from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

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
    - User ID from Firebase
    - Email and verification status
    - Role and permissions from Firestore
    - Additional user metadata
    
    The middleware handles:
    - Token verification
    - Firestore record creation (if needed)
    - User context enrichment
    
    Args:
        request: FastAPI request object
        credentials: Optional bearer token
        
    Returns:
        UserContext with complete user information
        
    Raises:
        HTTPException: 401 if not authenticated
    """
    # Debug logging to understand the auth flow
    logger.info("🔍 get_authenticated_user called for path: %s", request.url.path)
    logger.info("🔍 Request headers: %s", dict(request.headers))
    logger.info("🔍 Request state attributes: %s", dir(request.state))
    logger.info("🔍 Request scope keys: %s", list(request.scope.keys()))
    
    # Check if middleware has set user context - try multiple locations due to BaseHTTPMiddleware issues
    user_context = None
    
    # First try request.state (preferred)
    if hasattr(request.state, "user") and request.state.user is not None:
        user_context = request.state.user
        logger.info("🔍 Found user in request.state")
    # Then try request.scope (fallback for BaseHTTPMiddleware issues)
    elif "user" in request.scope and request.scope["user"] is not None:
        user_context = request.scope["user"]
        logger.info("🔍 Found user in request.scope (BaseHTTPMiddleware workaround)")
    # Try request attribute as last resort
    elif hasattr(request, "_auth_user") and request._auth_user is not None:
        user_context = request._auth_user
        logger.info("🔍 Found user in request._auth_user (BaseHTTPMiddleware workaround #2)")
    else:
        logger.warning("No user context in request.state, request.scope, or request._auth_user for path: %s", request.url.path)
        logger.warning("🔍 Auth header present: %s", "Authorization" in request.headers)
        logger.warning("🔍 request.state has attributes: %s", [attr for attr in dir(request.state) if not attr.startswith('_')])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.info("🔍 User context found: %s (type: %s)", user_context, type(user_context))
    
    if not isinstance(user_context, UserContext):
        logger.error("Invalid user context type: %s", type(user_context))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid authentication state",
        )

    logger.info("✅ Returning user context for user: %s", user_context.user_id)
    return user_context


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
    # Try request.state first
    if hasattr(request.state, "user") and request.state.user is not None:
        user_context = request.state.user
        if isinstance(user_context, UserContext):
            return user_context
    
    # Try request.scope as fallback
    if "user" in request.scope and request.scope["user"] is not None:
        user_context = request.scope["user"]
        if isinstance(user_context, UserContext):
            return user_context
    
    return None


# Type aliases for cleaner function signatures
AuthenticatedUser = Annotated[UserContext, Depends(get_authenticated_user)]
OptionalUser = Annotated[UserContext | None, Depends(get_optional_user)]


# Specialized dependencies for specific requirements
async def require_verified_email(
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


async def require_active_account(
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
        email_verified=context.is_verified,
        firebase_token="",  # Not available from context
        firebase_token_exp=None,
        created_at=context.created_at,
        last_login=context.last_login,
        profile=None,
    )


def get_websocket_user(token: str, request: Request) -> UserContext:
    """Get authenticated user for WebSocket connections.
    
    WebSocket connections pass the token as a query parameter rather than
    in headers, so we need special handling.
    
    Args:
        token: Firebase ID token from query parameter
        request: FastAPI request object (for state access)
        
    Returns:
        UserContext with complete user information
        
    Raises:
        HTTPException: 401 if token is invalid
    """
    # Set a fake authorization header for the middleware
    request.headers._list.append((b"authorization", f"Bearer {token}".encode()))

    # The middleware will process this and set request.state.user
    # We can then retrieve it
    if not hasattr(request.state, "user") or request.state.user is None:
        logger.warning("WebSocket authentication failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )

    return cast(UserContext, request.state.user)
