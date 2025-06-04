"""Firebase authentication utilities for WebSocket and HTTP endpoints."""

import logging
from typing import Any

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
    try:
        firebase_admin.get_app()
    except ValueError:
        # No default app exists, initialize it
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
except Exception:
    logger.exception("Failed to initialize Firebase Admin SDK")
    # In development, continue without Firebase

# Security scheme for API endpoints
security = HTTPBearer(auto_error=False)


def _raise_admin_required_error() -> None:
    """Raise admin privileges required error."""
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Admin privileges required",
    )


def get_current_user(
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
        return User(
            uid=decoded_token["uid"],
            email=decoded_token.get("email"),
            display_name=decoded_token.get("name"),
            email_verified=decoded_token.get("email_verified", False),
            firebase_token=credentials.credentials,
            created_at=None,
            last_login=None,
            profile=None,
        )

    except (auth.InvalidIdTokenError, auth.ExpiredIdTokenError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except Exception as e:
        logger.exception("Error verifying Firebase token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


def get_current_user_required(
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


def get_current_user_websocket(token: str) -> User:
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
        return User(
            uid=decoded_token["uid"],
            email=decoded_token.get("email"),
            display_name=decoded_token.get("name"),
            email_verified=decoded_token.get("email_verified", False),
            firebase_token=token,
            created_at=None,
            last_login=None,
            profile=None,
        )

    except (auth.InvalidIdTokenError, auth.ExpiredIdTokenError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
        ) from e
    except Exception as e:
        logger.exception("Error verifying WebSocket Firebase token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed"
        ) from e


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
        return User(
            uid=decoded_token["uid"],
            email=decoded_token.get("email"),
            display_name=decoded_token.get("name"),
            email_verified=decoded_token.get("email_verified", False),
            firebase_token=token,
            created_at=None,
            last_login=None,
            profile=None,
        )

    except (auth.InvalidIdTokenError, auth.ExpiredIdTokenError, ValueError):
        # Common auth/token parsing errors
        return None
    except (auth.RevokedIdTokenError, auth.CertificateFetchError, AttributeError) as e:
        logger.debug("Could not extract user from request: %s", e)
        return None


def verify_admin_user(user: User = Depends(get_current_user_required)) -> User:
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
            _raise_admin_required_error()

    except auth.UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
        ) from e
    except Exception as e:
        logger.exception("Error verifying admin user")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error verifying admin privileges",
        ) from e
    else:
        return user


def create_custom_token(
    uid: str, additional_claims: dict[str, Any] | None = None
) -> str:
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
    except Exception:
        logger.exception("Error creating custom token")
        raise


def set_custom_user_claims(uid: str, custom_claims: dict[str, Any]) -> None:
    """Set custom claims for a user.

    Args:
        uid: User ID
        custom_claims: Dictionary of custom claims to set
    """
    try:
        auth.set_custom_user_claims(uid, custom_claims)
        logger.info("Custom claims set for user %s: %s", uid, custom_claims)
    except Exception:
        logger.exception("Error setting custom claims for user %s", uid)
        raise


def get_user_by_email(email: str) -> auth.UserRecord | None:
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
    except Exception:
        logger.exception("Error getting user by email %s", email)
        raise


def delete_user(uid: str) -> None:
    """Delete a user account.

    Args:
        uid: User ID to delete
    """
    try:
        auth.delete_user(uid)
        logger.info("User %s deleted successfully", uid)
    except Exception:
        logger.exception("Error deleting user %s", uid)
        raise
