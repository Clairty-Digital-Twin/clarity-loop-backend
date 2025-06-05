"""Firebase authentication middleware and provider classes."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
import logging
import time
from typing import Any

from fastapi import FastAPI, Request, Response
from firebase_admin import auth as firebase_auth
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from clarity.models.auth import AuthError, Permission, UserContext, UserRole
from clarity.models.user import User
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
        self._user_cache: dict[str, User] = {}

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
        # Check if path is exempt from authentication
        if self._is_exempt_path(request.url.path):
            return await call_next(request)

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
        if not auth_header:
            raise AuthError(
                message="Missing Authorization header",
                status_code=401,
                error_code="missing_token",
            )

        if not auth_header.startswith("Bearer "):
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
            middleware_config: Middleware configuration options (dict or MiddlewareConfig)
        """
        self.credentials_path = credentials_path
        self.project_id = project_id

        config_dict: dict[str, Any]
        if middleware_config is None:
            config_dict = {}
        elif hasattr(middleware_config, "__dict__") and not isinstance(
            middleware_config, dict
        ):
            # It's likely a Pydantic model or similar object, access its dict representation
            config_dict = middleware_config.__dict__
        elif isinstance(middleware_config, dict):
            # It's already a dict
            config_dict = middleware_config
        else:
            # Fallback if it's some other type, though a dict is expected
            config_dict = {}

        self.middleware_config = config_dict  # Store the resolved config_dict
        self._initialized = False

        # Caching attributes for FirebaseAuthProvider itself
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
            # Perform any async initialization here
            await asyncio.sleep(0.1)  # Placeholder for actual initialization
            self._initialized = True
            logger.info("Firebase Admin SDK initialized successfully.")
        except Exception:
            logger.exception("Failed to initialize Firebase auth provider")
            raise

    def _remove_expired_tokens(self) -> None:
        """Remove expired tokens from the cache based on TTL."""
        if not self.cache_is_enabled:
            return
        current_time = time.time()
        expired_tokens = [
            t for t, data in self._token_cache.items()
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
                oldest_token = min(self._token_cache.items(), key=lambda item: item[1]["timestamp"])[0]
                del self._token_cache[oldest_token]
                logger.debug("Evicted oldest token %s to meet cache size limits.", oldest_token[:10])
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
            # Update its timestamp to mark it as recently used if implementing LRU behavior on top.
            # For now, just return it.
            # self._token_cache[token]["timestamp"] = time.time() # Optional: Update timestamp for LRU
            return self._token_cache[token]["user_data"]

        try:
            decoded_token = firebase_auth.verify_id_token(token, check_revoked=True)
            user_model_instance = User(
                uid=decoded_token["uid"],
                email=decoded_token.get("email"),
                display_name=decoded_token.get("name"),
                email_verified=decoded_token.get("email_verified", False),
                firebase_token=token,
                firebase_token_exp=decoded_token.get("exp"),
                created_at=(
                    datetime.fromtimestamp(decoded_token.get("auth_time"))
                    if decoded_token.get("auth_time")
                    else None
                ),
                last_login=None,
                profile=None,
            )
            user_data_dict = user_model_instance.model_dump()

            if self.cache_is_enabled:
                # Ensure there's space if the cache is at or over its limit
                if len(self._token_cache) >= self._token_cache_max_size:
                    # Need to make space for 1 new item, so target max_size - 1 before adding
                    self._evict_oldest_to_target_count(target_count=self._token_cache_max_size - 1)

                self._token_cache[token] = {"user_data": user_data_dict, "timestamp": time.time()}
            return user_data_dict
        except firebase_auth.RevokedIdTokenError:
            logger.warning("Revoked Firebase ID token received: %s", token[:20] + "...")
            return None
        except firebase_auth.UserDisabledError:
            logger.warning(
                "Disabled user tried to authenticate: %s", token[:20] + "..."
            )
            return None
        except firebase_auth.InvalidIdTokenError:
            logger.warning("Invalid Firebase ID token: %s", token[:20] + "...")
            return None
        except Exception as e:
            logger.exception("Error verifying Firebase token: %s", e)
            return None

    async def create_user(self, user_data: dict[str, Any]) -> User:
        """Create a new Firebase user.

        Args:
            user_data: Dictionary containing user creation data (email, password, etc.)

        Returns:
            Created User object
        """
        if not self._initialized:
            await self.initialize()
        try:
            user_record = firebase_auth.create_user(**user_data)
            # Convert UserRecord to your User model
            return User(
                uid=user_record.uid,
                email=user_record.email,
                display_name=user_record.display_name,
                email_verified=user_record.email_verified,
                # Timestamps
                created_at=datetime.fromtimestamp(user_record.user_metadata.creation_timestamp / 1000) if user_record.user_metadata else None,  # type: ignore
                last_login=datetime.fromtimestamp(user_record.user_metadata.last_sign_in_timestamp / 1000) if user_record.user_metadata and user_record.user_metadata.last_sign_in_timestamp else None,  # type: ignore
            )
        except Exception as e:
            logger.exception("Error creating Firebase user: %s", e)
            raise AuthError(
                message=f"Failed to create user: {e}",
                status_code=500,
                error_code="user_creation_failed",
            ) from e

    async def get_user_info(self, user_id: str) -> dict[str, Any] | None:
        """Get user information by Firebase UID.

        Args:
            user_id: Firebase User ID (UID)

        Returns:
            User object if found, None otherwise
        """
        if not self._initialized:
            await self.initialize()
        try:
            user_record = firebase_auth.get_user(user_id)
            user = User(
                uid=user_record.uid,
                email=user_record.email,
                display_name=user_record.display_name,
                email_verified=user_record.email_verified,
                created_at=datetime.fromtimestamp(user_record.user_metadata.creation_timestamp / 1000) if user_record.user_metadata else None,  # type: ignore
                last_login=datetime.fromtimestamp(user_record.user_metadata.last_sign_in_timestamp / 1000) if user_record.user_metadata and user_record.user_metadata.last_sign_in_timestamp else None,  # type: ignore
            )
            return user.model_dump()
        except firebase_auth.UserNotFoundError:
            logger.debug("User not found with UID: %s", user_id)
            return None
        except Exception as e:
            logger.exception("Error fetching user info for UID %s: %s", user_id, e)
            return None

    async def cleanup(self) -> None:
        """Cleanup resources when shutting down."""
        self._token_cache.clear()
        logger.info("Firebase authentication provider cleanup complete")
        self._initialized = False
