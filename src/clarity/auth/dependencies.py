"""Authentication dependencies for Clean Architecture.

This module provides the single source of truth for authentication dependencies
following Robert C. Martin's principles and security best practices.
"""

import logging
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from clarity.models.auth import UserContext
from clarity.models.user import User

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)


async def get_authenticated_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
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
    # Check if middleware has set user context
    if not hasattr(request.state, "user") or request.state.user is None:
        logger.warning("No user context in request.state for path: %s", request.url.path)
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


async def get_optional_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
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