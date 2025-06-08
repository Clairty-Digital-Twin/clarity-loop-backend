"""Firebase authentication middleware and provider classes."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
import logging
import time
from typing import Any, cast

from fastapi import FastAPI, Request, Response
from firebase_admin import auth as firebase_auth
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from clarity.models.auth import AuthError, Permission, UserContext, UserRole
from clarity.ports.auth_ports import IAuthProvider

logger = logging.getLogger(__name__)


class FirebaseAuthMiddleware(BaseHTTPMiddleware):
    """Firebase authentication middleware for FastAPI applications."""

    def __init__(
        self,
        app: FastAPI,
        auth_provider: IAuthProvider,
        exempt_paths: list[str] | None = None,
        *,
        cache_enabled: bool = True,
        graceful_degradation: bool = True,
    ) -> None:
        """Initialize Firebase authentication middleware.

        Args:
            app: FastAPI application instance
            auth_provider: Authentication provider instance
            exempt_paths: List of paths to exempt from authentication
            cache_enabled: Whether to enable authentication caching
            graceful_degradation: Whether to continue on auth failures
        """
        super().__init__(app)
        self.auth_provider = auth_provider
        self.exempt_paths = exempt_paths or [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/metrics",
            "/api/v1/ws",  # WebSocket endpoints handle auth separately
        ]
        self.cache_enabled = cache_enabled
        self.graceful_degradation = graceful_degradation
        self._user_cache: dict[str, UserContext] = {}

        logger.info("Firebase authentication middleware initialized")
        logger.info("Exempt paths: %s", self.exempt_paths)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process request through authentication middleware.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            HTTP response
        """
        # CRITICAL DEBUG: Log EVERY request that hits middleware
        logger.warning("🔥 MIDDLEWARE HIT: %s %s", request.method, request.url.path)
        
        # Check if path is exempt from authentication
        if self._is_exempt_path(request.url.path):
            logger.warning("🔥 PATH IS EXEMPT: %s", request.url.path)
            return await call_next(request)
        
        logger.warning("🔥 PATH REQUIRES AUTH: %s", request.url.path)

        # Try to authenticate the request
        try:
            user_context = await self._authenticate_request(request)
            request.state.user = user_context
        except AuthError as e:
            # Handle authentication errors
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": e.error_code,
                    "message": e.message,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Authentication failed: %s", e)
            if not self.graceful_degradation:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "authentication_failed",
                        "message": "Authentication required",
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )
            # For graceful degradation, set user as None
            request.state.user = None

        return await call_next(request)

    def _is_exempt_path(self, path: str) -> bool:
        """Check if a path is exempt from authentication.

        Args:
            path: Request path to check

        Returns:
            True if path is exempt
        """
        return any(path.startswith(exempt_path) for exempt_path in self.exempt_paths)

    @staticmethod
    def _extract_token(request: Request) -> str:
        """Extract Firebase token from Authorization header.

        Args:
            request: HTTP request to extract token from

        Returns:
            Firebase ID token

        Raises:
            AuthError: If token is missing or invalid format
        """
        auth_header = request.headers.get("Authorization")
        logger.warning("🔍 Auth header seen: %s", auth_header)
        
        if not auth_header:
            logger.error("❌ MISSING Authorization header")
            raise AuthError(
                message="Missing Authorization header",
                status_code=401,
                error_code="missing_token",
            )

        if not auth_header.startswith("Bearer "):
            logger.error("❌ INVALID Authorization header format: %s", auth_header[:50])
            raise AuthError(
                message="Invalid Authorization header format",
                status_code=401,
                error_code="invalid_token_format",
            )

        token = auth_header[7:]  # Remove "Bearer " prefix
        if not token:
            raise AuthError(
                message="Empty token in Authorization header",
                status_code=401,
                error_code="empty_token",
            )

        return token

    @staticmethod
    def _create_user_context(user_info: dict[str, Any]) -> UserContext:
        """Create user context from user information.

        Args:
            user_info: Dictionary containing user information

        Returns:
            UserContext object with user details and permissions
        """
        # Extract user role from custom claims or roles list
        roles = user_info.get("roles", [])
        if "admin" in roles:
            role = UserRole.ADMIN
        elif "clinician" in roles:
            role = UserRole.CLINICIAN
        else:
            role = UserRole.PATIENT  # Default to patient

        # Set permissions based on role
        permissions = set()
        if role == UserRole.ADMIN:
            permissions = {
                Permission.SYSTEM_ADMIN,
                Permission.MANAGE_USERS,
                Permission.READ_OWN_DATA,
                Permission.WRITE_OWN_DATA,
                Permission.READ_PATIENT_DATA,
                Permission.WRITE_PATIENT_DATA,
                Permission.READ_ANONYMIZED_DATA,
            }
        elif role == UserRole.CLINICIAN:
            permissions = {
                Permission.READ_OWN_DATA,
                Permission.WRITE_OWN_DATA,
                Permission.READ_PATIENT_DATA,
                Permission.WRITE_PATIENT_DATA,
            }
        else:  # PATIENT
            permissions = {Permission.READ_OWN_DATA, Permission.WRITE_OWN_DATA}

        return UserContext(
            user_id=user_info["user_id"],
            email=user_info["email"],
            role=role,
            permissions=list(permissions),
            is_verified=user_info.get("verified", False),
            custom_claims=user_info.get("custom_claims", {}),
            created_at=user_info.get("created_at"),
            last_login=user_info.get("last_login"),
        )

    async def _authenticate_request(self, request: Request) -> UserContext:
        """Authenticate a request and return user context.

        Args:
            request: HTTP request to authenticate

        Returns:
            UserContext object if authenticated

        Raises:
            AuthError: If authentication fails
        """
        # Extract token from request
        token = self._extract_token(request)

        # Verify token with auth provider
        if not self.auth_provider:
            raise AuthError(
                message="Authentication provider not configured",
                status_code=500,
                error_code="auth_provider_not_configured",
            )

        user_info = await self.auth_provider.verify_token(token)
        if not user_info:
            raise AuthError(
                message="Invalid or expired token",
                status_code=401,
                error_code="invalid_token",
            )

        # At this point, user_info is guaranteed to be a dict[str, Any]
        # The previous block trying to handle User object or dict was causing issues.
        return self._create_user_context(user_info)


class FirebaseAuthProvider(IAuthProvider):
    """Firebase authentication provider.

    Handles token verification and user information retrieval using Firebase Admin SDK.
    """

    def __init__(
        self,
        credentials_path: str | None = None,
        project_id: str | None = None,
        middleware_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize Firebase authentication provider.

        Args:
            credentials_path: Path to Firebase service account credentials
            project_id: Firebase project ID
            middleware_config: Middleware configuration options (already a dict or None)
        """
        self.credentials_path = credentials_path
        self.project_id = project_id

        # middleware_config is now expected to be a dict or None from the container
        config_dict: dict[str, Any] = (
            middleware_config if middleware_config is not None else {}
        )

        self.middleware_config = config_dict  # Store the resolved config_dict
        self._initialized = False

        # Caching attributes for FirebaseAuthProvider itself
        # Get auth_provider_config from the passed config_dict, or default to empty dict
        auth_provider_specific_config = self.middleware_config.get(
            "auth_provider_config", {}
        )
        self.cache_is_enabled = auth_provider_specific_config.get("cache_enabled", True)
        self._token_cache_ttl_seconds = auth_provider_specific_config.get(
            "cache_ttl_seconds", 300
        )  # 5 minutes
        self._token_cache_max_size = auth_provider_specific_config.get(
            "cache_max_size", 1000
        )
        self._token_cache: dict[str, dict[str, Any]] = (
            {}
        )  # token -> {"user_data": dict, "timestamp": float}

        logger.info("Firebase Authentication Provider initialized.")
        if credentials_path:
            logger.info("Using credentials from: %s", credentials_path)
        if project_id:
            logger.info("Firebase project ID: %s", project_id)

    async def initialize(self) -> None:
        """Initialize Firebase Admin SDK if not already initialized."""
        if self._initialized:
            return

        try:
            # Initialize Firebase Admin SDK if not already initialized
            import firebase_admin
            from firebase_admin import credentials

            try:
                firebase_admin.get_app()
                logger.info("Firebase Admin SDK already initialized")
            except ValueError:
                # No default app exists, initialize it
                logger.info("Initializing Firebase Admin SDK...")

                # Try to use credentials from environment first
                import os

                if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                    # Use Application Default Credentials (file path set by main.py)
                    cred = credentials.ApplicationDefault()
                    firebase_admin.initialize_app(cred)
                    logger.info(
                        "Firebase Admin SDK initialized with Application Default Credentials"
                    )
                elif self.credentials_path:
                    # Use provided credentials path
                    cred = credentials.Certificate(self.credentials_path)
                    firebase_admin.initialize_app(cred)
                    logger.info(
                        "Firebase Admin SDK initialized with credentials from: %s",
                        self.credentials_path,
                    )
                else:
                    # Try to initialize with project ID only (for emulator/local dev)
                    firebase_admin.initialize_app()
                    logger.info("Firebase Admin SDK initialized with default settings")

            self._initialized = True
            logger.info("✅ Firebase Admin SDK ready for token verification")
        except Exception:
            logger.exception("Failed to initialize Firebase auth provider")
            raise

    def _remove_expired_tokens(self) -> None:
        """Remove expired tokens from the cache based on TTL."""
        if not self.cache_is_enabled:
            return
        current_time = time.time()
        expired_tokens = [
            t
            for t, data in self._token_cache.items()
            if current_time - data["timestamp"] > self._token_cache_ttl_seconds
        ]
        for t in expired_tokens:
            if t in self._token_cache:
                del self._token_cache[t]
                logger.debug("Removed expired token %s from cache.", t[:10])

    def _evict_oldest_to_target_count(self, target_count: int) -> None:
        """Evict the oldest items until the cache size is at or below target_count."""
        if not self.cache_is_enabled:
            return
        while len(self._token_cache) > target_count:
            if not self._token_cache:  # Should not happen if len > target_count
                break
            try:
                # Find and remove the oldest entry (smallest timestamp)
                oldest_token = min(
                    self._token_cache.items(), key=lambda item: item[1]["timestamp"]
                )[0]
                del self._token_cache[oldest_token]
                logger.debug(
                    "Evicted oldest token %s to meet cache size limits.",
                    oldest_token[:10],
                )
            except ValueError:  # Cache became empty during removal
                break

    async def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify Firebase ID token and return user information as a dictionary.

        Args:
            token: Firebase ID token

        Returns:
            User information dictionary if token is valid, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        self._remove_expired_tokens()  # Always try to remove expired tokens first

        # Check cache first if enabled
        if self.cache_is_enabled and token in self._token_cache:
            # Item is in cache and not expired (since _remove_expired_tokens was called)
            logger.info("🔵 Token found in cache")
            return cast("dict[str, Any]", self._token_cache[token]["user_data"])

        logger.info("🔍 Attempting to verify Firebase token (length: %d)", len(token))
        logger.debug("🔍 Token preview: %s...%s", token[:20], token[-20:])
        
        try:
            # Log current Firebase app state
            app = firebase_auth.get_app()
            logger.info("🔥 Firebase app name: %s", app.name)
            logger.info("🔥 Firebase project_id: %s", app.project_id if hasattr(app, 'project_id') else 'unknown')
            
            # TEMPORARY DEBUG: Try without revocation check first
            logger.info("🧪 DEBUGGING: Attempting token verification WITHOUT revocation check...")
            try:
                decoded_token_no_revoke = firebase_auth.verify_id_token(token, check_revoked=False)
                logger.info("✅ Token verified successfully WITHOUT revocation check! UID: %s", decoded_token_no_revoke.get('uid'))
                
                # Now try with revocation check
                logger.info("🧪 DEBUGGING: Now attempting token verification WITH revocation check...")
                decoded_token = firebase_auth.verify_id_token(token, check_revoked=True)
                logger.info("✅ Token verified successfully WITH revocation check! UID: %s", decoded_token.get('uid'))
                
            except Exception as revoke_check_error:
                logger.error("❌ DEBUGGING: Revocation check failed: %s", str(revoke_check_error))
                logger.error("❌ DEBUGGING: Error type: %s", type(revoke_check_error).__name__)
                logger.exception("❌ DEBUGGING: Full revocation check error:")
                
                # Fall back to token without revocation check for now
                logger.warning("⚠️ DEBUGGING: Using token WITHOUT revocation check as fallback")
                decoded_token = decoded_token_no_revoke

            # Extract custom claims to determine roles
            custom_claims = decoded_token.get("custom_claims", {})
            roles = []
            if custom_claims.get("admin"):
                roles.append("admin")
            if custom_claims.get("clinician"):
                roles.append("clinician")

            # Create user data dict in the format expected by _create_user_context
            user_data_dict = {
                "user_id": decoded_token["uid"],
                "email": decoded_token.get("email"),
                "verified": decoded_token.get("email_verified", False),
                "roles": roles,
                "custom_claims": custom_claims,
                "created_at": (
                    datetime.fromtimestamp(decoded_token.get("auth_time"), tz=UTC)
                    if decoded_token.get("auth_time")
                    else None
                ),
                "last_login": None,
            }

            if self.cache_is_enabled:
                if len(self._token_cache) >= self._token_cache_max_size:
                    self._evict_oldest_to_target_count(
                        target_count=self._token_cache_max_size - 1
                    )
                self._token_cache[token] = {
                    "user_data": user_data_dict,
                    "timestamp": time.time(),
                }
            return user_data_dict  # noqa: TRY300 - Return happens regardless of caching, if block is for side-effect
        except firebase_auth.RevokedIdTokenError as e:
            logger.error("❌ Revoked Firebase ID token: %s", str(e))
            logger.error("Token preview: %s...%s", token[:20], token[-20:])
            return None
        except firebase_auth.UserDisabledError as e:
            logger.error("❌ Disabled user tried to authenticate: %s", str(e))
            return None
        except firebase_auth.InvalidIdTokenError as e:
            logger.error("❌ Invalid Firebase ID token: %s", str(e))
            logger.error("Token preview: %s...%s", token[:20], token[-20:])
            return None
        except firebase_auth.ExpiredIdTokenError as e:
            logger.error("❌ Expired Firebase ID token: %s", str(e))
            return None
        except firebase_auth.CertificateFetchError as e:
            logger.error("❌ Certificate fetch error: %s", str(e))
            return None
        except Exception as e:
            logger.error("❌ Unexpected error verifying Firebase token: %s", type(e).__name__)
            logger.error("Error details: %s", str(e))
            logger.exception("Full exception details:")
            return None

    async def get_user_info(self, user_id: str) -> dict[str, Any] | None:
        """Get user information by Firebase UID.

        Args:
            user_id: Firebase User ID (UID)

        Returns:
            User information dictionary if found, None otherwise
        """
        if not self._initialized:
            await self.initialize()
        try:
            user_record = firebase_auth.get_user(user_id)

            # Extract custom claims to determine roles
            custom_claims = user_record.custom_claims or {}
            roles = []
            if custom_claims.get("admin"):
                roles.append("admin")
            if custom_claims.get("clinician"):
                roles.append("clinician")

            # Return user data in the format expected by _create_user_context
            return {
                "user_id": user_record.uid,
                "email": user_record.email,
                "verified": user_record.email_verified,
                "roles": roles,
                "custom_claims": custom_claims,
                "created_at": (
                    datetime.fromtimestamp(
                        user_record.user_metadata.creation_timestamp / 1000, tz=UTC
                    )
                    if user_record.user_metadata
                    else None
                ),
                "last_login": (
                    datetime.fromtimestamp(
                        user_record.user_metadata.last_sign_in_timestamp / 1000, tz=UTC
                    )
                    if user_record.user_metadata
                    and user_record.user_metadata.last_sign_in_timestamp
                    else None
                ),
            }
        except firebase_auth.UserNotFoundError:
            logger.debug("User not found with UID: %s", user_id)
            return None
        except Exception:  # Keep broad for unknown Firebase/network issues
            logger.exception("Error fetching user info for UID %s", user_id)
            return None

    async def cleanup(self) -> None:
        """Cleanup resources when shutting down."""
        self._token_cache.clear()
        logger.info("Firebase authentication provider cleanup complete")
        self._initialized = False
