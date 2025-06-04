"""Firebase authentication utilities for WebSocket and HTTP endpoints."""

import logging
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import firebase_admin
from firebase_admin import auth, credentials

from clarity.core.config import get_settings
from clarity.models.user import User

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize Firebase Admin SDK
try:
    if not firebase_admin._apps:
        # In production, use service account key
        # In development, you can use default credentials
        if (
            hasattr(settings, "FIREBASE_SERVICE_ACCOUNT_KEY")
            and settings.FIREBASE_SERVICE_ACCOUNT_KEY
        ):
            cred = credentials.Certificate(
                getattr(settings, "FIREBASE_SERVICE_ACCOUNT_KEY", None)
            )
        else:
            # Use default credentials (useful for local development)
            cred = credentials.ApplicationDefault()

        firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
    # In development, continue without Firebase

# Security scheme for API endpoints
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> User | None:
    """Get current authenticated user from Firebase token.

    Returns None if no valid token is provided (for optional authentication).
    Raises HTTPException for invalid tokens.
    """
    if not credentials:
        return None

    try:
        # Verify the Firebase ID token
        decoded_token = auth.verify_id_token(credentials.credentials)

        # Create user object from token
        user = User(
            uid=decoded_token["uid"],
            email=decoded_token.get("email"),
            display_name=decoded_token.get("name"),
            email_verified=decoded_token.get("email_verified", False),
            firebase_token=credentials.credentials,
            created_at=None,
            last_login=None,
            profile=None,
        )

        return user

    except (auth.InvalidIdTokenError, auth.ExpiredIdTokenError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Error verifying Firebase token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_required(
    user: User | None = Depends(get_current_user),
) -> User:
    """Get current authenticated user (required).

    Raises HTTPException if no valid user is authenticated.
    """
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_current_user_websocket(token: str) -> User:
    """Get current authenticated user from WebSocket token parameter.

    Args:
        token: Firebase ID token from WebSocket query parameter

    Returns:
        User object

    Raises:
        HTTPException: If token is invalid
    """
    try:
        # Verify the Firebase ID token
        decoded_token = auth.verify_id_token(token)

        # Create user object from token
        user = User(
            uid=decoded_token["uid"],
            email=decoded_token.get("email"),
            display_name=decoded_token.get("name"),
            email_verified=decoded_token.get("email_verified", False),
            firebase_token=token,
            created_at=None,
            last_login=None,
            profile=None,
        )

        return user

    except (auth.InvalidIdTokenError, auth.ExpiredIdTokenError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
        )
    except Exception as e:
        logger.error(f"Error verifying WebSocket Firebase token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed"
        )


def get_user_from_request(request: Request) -> User | None:
    """Extract user from request headers (for middleware use).

    Args:
        request: FastAPI request object

    Returns:
        User object if authenticated, None otherwise
    """
    authorization = request.headers.get("Authorization")
    if not authorization:
        return None

    try:
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() != "bearer":
            return None

        # Verify the Firebase ID token
        decoded_token = auth.verify_id_token(token)

        # Create user object from token
        user = User(
            uid=decoded_token["uid"],
            email=decoded_token.get("email"),
            display_name=decoded_token.get("name"),
            email_verified=decoded_token.get("email_verified", False),
            firebase_token=token,
            created_at=None,
            last_login=None,
            profile=None,
        )

        return user

    except Exception as e:
        logger.debug(f"Could not extract user from request: {e}")
        return None


async def verify_admin_user(user: User = Depends(get_current_user_required)) -> User:
    """Verify that the current user has admin privileges.

    This is a placeholder implementation. In a real application,
    you would check user roles/claims from Firebase Custom Claims
    or your own user management system.
    """
    try:
        # Get user record to check custom claims
        user_record = auth.get_user(user.uid)

        # Check if user has admin claim
        custom_claims = user_record.custom_claims or {}
        if not custom_claims.get("admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required",
            )

        return user

    except auth.UserNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
        )
    except Exception as e:
        logger.error(f"Error verifying admin user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error verifying admin privileges",
        )


def create_custom_token(uid: str, additional_claims: dict | None = None) -> str:
    """Create a custom Firebase token for a user.

    This is useful for server-side user creation or testing.

    Args:
        uid: User ID
        additional_claims: Additional claims to include in the token

    Returns:
        Custom Firebase token
    """
    try:
        return auth.create_custom_token(uid, additional_claims)
    except Exception as e:
        logger.error(f"Error creating custom token: {e}")
        raise


def set_custom_user_claims(uid: str, custom_claims: dict):
    """Set custom claims for a user.

    Args:
        uid: User ID
        custom_claims: Dictionary of custom claims to set
    """
    try:
        auth.set_custom_user_claims(uid, custom_claims)
        logger.info(f"Custom claims set for user {uid}: {custom_claims}")
    except Exception as e:
        logger.error(f"Error setting custom claims for user {uid}: {e}")
        raise


def get_user_by_email(email: str):
    """Get user record by email.

    Args:
        email: User email address

    Returns:
        Firebase UserRecord
    """
    try:
        return auth.get_user_by_email(email)
    except auth.UserNotFoundError:
        return None
    except Exception as e:
        logger.error(f"Error getting user by email {email}: {e}")
        raise


def delete_user(uid: str):
    """Delete a user account.

    Args:
        uid: User ID to delete
    """
    try:
        auth.delete_user(uid)
        logger.info(f"User {uid} deleted successfully")
    except Exception as e:
        logger.error(f"Error deleting user {uid}: {e}")
        raise
