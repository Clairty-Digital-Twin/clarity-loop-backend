"""Firebase authentication utilities for WebSocket and HTTP endpoints."""

import logging
import os
from typing import Any, cast

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import firebase_admin
from firebase_admin import auth, credentials

from clarity.core.config import get_settings
from clarity.models.auth import UserContext
from clarity.models.user import User

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize Firebase Admin SDK
try:
    try:
        app = firebase_admin.get_app()
        logger.info("🔵 Firebase Admin SDK already initialized with app: %s", app.name)
    except ValueError:
        # No default app exists, initialize it
        logger.info("🔄 Initializing Firebase Admin SDK...")

        # Check various credential sources
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            logger.info(
                "🔑 Using GOOGLE_APPLICATION_CREDENTIALS from: %s",
                os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
            )
            cred = credentials.ApplicationDefault()
        elif (
            hasattr(settings, "firebase_credentials_path")
            and settings.firebase_credentials_path
        ):
            logger.info(
                "🔑 Using firebase_credentials_path from settings: %s",
                settings.firebase_credentials_path,
            )
            cred = credentials.Certificate(settings.firebase_credentials_path)
        else:
            logger.warning("⚠️ No credentials found, using ApplicationDefault")
            cred = credentials.ApplicationDefault()

        app = firebase_admin.initialize_app(cred)
        logger.info(
            "✅ Firebase Admin SDK initialized successfully with app: %s", app.name
        )
except Exception:
    logger.exception("🔥 Failed to initialize Firebase Admin SDK")
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
        # TEMPORARY DEBUG: Add detailed logging for token verification
        logger.info("🔍 DEBUGGING get_current_user: Attempting token verification (length: %d)", len(credentials.credentials))
        logger.debug("🔍 DEBUGGING get_current_user: Token preview: %s...%s", credentials.credentials[:20], credentials.credentials[-20:])

        # Try without revocation check first
        logger.info("🧪 DEBUGGING get_current_user: Attempting WITHOUT revocation check...")
        try:
            decoded_token_no_revoke = auth.verify_id_token(
                credentials.credentials,
                check_revoked=False,
                clock_skew_seconds=30,
            )
            logger.info("✅ DEBUGGING get_current_user: Token verified WITHOUT revocation check! UID: %s", decoded_token_no_revoke.get('uid'))

            # Now try with revocation check
            logger.info("🧪 DEBUGGING get_current_user: Now attempting WITH revocation check...")
            decoded_token = auth.verify_id_token(
                credentials.credentials,
                check_revoked=True,  # Prevent authentication bypass with stolen tokens
                clock_skew_seconds=30,  # Allow for minor clock drift
            )
            logger.info("✅ DEBUGGING get_current_user: Token verified WITH revocation check! UID: %s", decoded_token.get('uid'))

        except Exception as revoke_check_error:
            logger.error("❌ DEBUGGING get_current_user: Revocation check failed: %s", str(revoke_check_error))
            logger.error("❌ DEBUGGING get_current_user: Error type: %s", type(revoke_check_error).__name__)
            logger.exception("❌ DEBUGGING get_current_user: Full revocation check error:")

            # Fall back to token without revocation check for now
            logger.warning("⚠️ DEBUGGING get_current_user: Using token WITHOUT revocation check as fallback")
            decoded_token = decoded_token_no_revoke

        # Create user object from token
        return User(
            uid=decoded_token["uid"],
            email=decoded_token.get("email"),
            display_name=decoded_token.get("name"),
            email_verified=decoded_token.get("email_verified", False),
            firebase_token=credentials.credentials,
            firebase_token_exp=decoded_token.get("exp"),
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


def get_current_user_context(
    request: Request,
) -> UserContext | None:
    """Get current authenticated user context from middleware.

    Returns None if no valid user context is available (for optional authentication).
    """
    if not hasattr(request.state, "user"):
        return None

    return cast("UserContext | None", request.state.user)


def get_current_user_context_required(
    request: Request,
) -> UserContext:
    """Get current authenticated user context (required).

    This function works with the FirebaseAuthMiddleware which sets
    request.state.user to a UserContext object.

    Raises HTTPException if no valid user context is authenticated.
    """
    logger.warning("🔍 get_current_user_context_required called")
    logger.warning("🔍 Request path: %s", request.url.path)
    logger.warning("🔍 Has request.state: %s", hasattr(request, "state"))
    if hasattr(request, "state"):
        logger.warning("🔍 Has request.state.user: %s", hasattr(request.state, "user"))
        if hasattr(request.state, "user"):
            logger.warning("🔍 request.state.user value: %s", request.state.user)

    if not hasattr(request.state, "user") or request.state.user is None:
        logger.error("❌ No user context found in request.state for path: %s", request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return cast("UserContext", request.state.user)


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
        # TEMPORARY DEBUG: Add detailed logging for WebSocket token verification
        logger.info("🔍 DEBUGGING get_current_user_websocket: Attempting token verification (length: %d)", len(token))
        logger.debug("🔍 DEBUGGING get_current_user_websocket: Token preview: %s...%s", token[:20], token[-20:])

        # Try without revocation check first
        logger.info("🧪 DEBUGGING get_current_user_websocket: Attempting WITHOUT revocation check...")
        try:
            decoded_token_no_revoke = auth.verify_id_token(token, check_revoked=False)
            logger.info("✅ DEBUGGING get_current_user_websocket: Token verified WITHOUT revocation check! UID: %s", decoded_token_no_revoke.get('uid'))

            # Now try with revocation check
            logger.info("🧪 DEBUGGING get_current_user_websocket: Now attempting WITH revocation check...")
            decoded_token = auth.verify_id_token(token, check_revoked=True)
            logger.info("✅ DEBUGGING get_current_user_websocket: Token verified WITH revocation check! UID: %s", decoded_token.get('uid'))

        except Exception as revoke_check_error:
            logger.error("❌ DEBUGGING get_current_user_websocket: Revocation check failed: %s", str(revoke_check_error))
            logger.error("❌ DEBUGGING get_current_user_websocket: Error type: %s", type(revoke_check_error).__name__)
            logger.exception("❌ DEBUGGING get_current_user_websocket: Full revocation check error:")

            # Fall back to token without revocation check for now
            logger.warning("⚠️ DEBUGGING get_current_user_websocket: Using token WITHOUT revocation check as fallback")
            decoded_token = decoded_token_no_revoke

        # Create user object from token
        return User(
            uid=decoded_token["uid"],
            email=decoded_token.get("email"),
            display_name=decoded_token.get("name"),
            email_verified=decoded_token.get("email_verified", False),
            firebase_token=token,
            firebase_token_exp=decoded_token.get("exp"),
            created_at=None,
            last_login=None,
            profile=None,
        )

    except (
        auth.InvalidIdTokenError,
        auth.ExpiredIdTokenError,
        auth.RevokedIdTokenError,
    ) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, expired, or revoked authentication token",
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

        # TEMPORARY DEBUG: Add detailed logging for request token verification
        logger.info("🔍 DEBUGGING get_user_from_request: Attempting token verification (length: %d)", len(token))
        logger.debug("🔍 DEBUGGING get_user_from_request: Token preview: %s...%s", token[:20], token[-20:])

        # Try without revocation check first
        logger.info("🧪 DEBUGGING get_user_from_request: Attempting WITHOUT revocation check...")
        try:
            decoded_token_no_revoke = auth.verify_id_token(token, check_revoked=False)
            logger.info("✅ DEBUGGING get_user_from_request: Token verified WITHOUT revocation check! UID: %s", decoded_token_no_revoke.get('uid'))

            # Now try with revocation check
            logger.info("🧪 DEBUGGING get_user_from_request: Now attempting WITH revocation check...")
            decoded_token = auth.verify_id_token(token, check_revoked=True)
            logger.info("✅ DEBUGGING get_user_from_request: Token verified WITH revocation check! UID: %s", decoded_token.get('uid'))

        except Exception as revoke_check_error:
            logger.error("❌ DEBUGGING get_user_from_request: Revocation check failed: %s", str(revoke_check_error))
            logger.error("❌ DEBUGGING get_user_from_request: Error type: %s", type(revoke_check_error).__name__)
            logger.exception("❌ DEBUGGING get_user_from_request: Full revocation check error:")

            # Fall back to token without revocation check for now
            logger.warning("⚠️ DEBUGGING get_user_from_request: Using token WITHOUT revocation check as fallback")
            decoded_token = decoded_token_no_revoke

        # Create user object from token
        return User(
            uid=decoded_token["uid"],
            email=decoded_token.get("email"),
            display_name=decoded_token.get("name"),
            email_verified=decoded_token.get("email_verified", False),
            firebase_token=token,
            firebase_token_exp=decoded_token.get("exp"),
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
        custom_token_bytes: bytes = auth.create_custom_token(uid, additional_claims)
        return custom_token_bytes.decode("utf-8")
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
