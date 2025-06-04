"""Firebase authentication middleware and provider classes."""

import asyncio
from collections.abc import Callable
import logging
import time
from typing import Any

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from clarity.auth.firebase_auth import get_user_from_request
from clarity.models.user import User

# Removed unused imports - using built-in types instead

logger = logging.getLogger(__name__)


class FirebaseAuthMiddleware(BaseHTTPMiddleware):
    """Firebase authentication middleware for FastAPI applications."""

    def __init__(
        self,
        app: FastAPI,
        auth_provider: Any,
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
        logger.info(f"Exempt paths: {self.exempt_paths}")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
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

        # Try to get authenticated user
        user = None
        if self.auth_provider:
            try:
                user = get_user_from_request(request)
                if user and self.cache_enabled:
                    # Cache user for this request
                    self._user_cache[user.uid] = user
            except Exception as e:
                logger.warning(f"Authentication failed: {e}")
                if not self.graceful_degradation:
                    # Return 401 if graceful degradation is disabled
                    from fastapi import HTTPException, status

                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required",
                    )

        # Add user to request state
        request.state.user = user

        return await call_next(request)

    def _is_exempt_path(self, path: str) -> bool:
        """Check if a path is exempt from authentication.

        Args:
            path: Request path to check

        Returns:
            True if path is exempt
        """
        for exempt_path in self.exempt_paths:
            if path.startswith(exempt_path):
                return True
        return False

    def _extract_token(self, request: Request) -> str:
        """Extract Firebase token from Authorization header.

        Args:
            request: HTTP request to extract token from

        Returns:
            Firebase ID token

        Raises:
            AuthError: If token is missing or invalid format
        """
        from clarity.models.auth import AuthError

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

    def _create_user_context(self, user_info: dict[str, Any]) -> Any:
        """Create user context from user information.

        Args:
            user_info: Dictionary containing user information

        Returns:
            UserContext object with user details and permissions
        """
        from clarity.models.auth import Permission, UserContext, UserRole

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
            permissions=permissions,
            verified=user_info.get("verified", False),
            custom_claims=user_info.get("custom_claims", {}),
        )


class FirebaseAuthProvider:
    """Firebase authentication provider for dependency injection."""

    def __init__(
        self,
        credentials_path: str | None = None,
        project_id: str | None = None,
        middleware_config: dict | Any | None = None,
    ) -> None:
        """Initialize Firebase authentication provider.

        Args:
            credentials_path: Path to Firebase service account credentials
            project_id: Firebase project ID
            middleware_config: Middleware configuration options (dict or MiddlewareConfig)
        """
        self.credentials_path = credentials_path
        self.project_id = project_id
        
        # Handle both dict and MiddlewareConfig objects
        config_dict: dict[str, Any] = {}
        if middleware_config is None:
            config_dict = {}
        elif hasattr(middleware_config, '__dict__'):
            # It's a MiddlewareConfig object, convert to dict
            config_dict = middleware_config.__dict__
        else:
            # It's already a dict
            config_dict = middleware_config
            
        self.middleware_config = config_dict
        self._initialized = False
        self._token_cache: dict[str, dict[str, Any]] = {}  # token -> {data, timestamp}
        self._cache_ttl = config_dict.get('cache_ttl_seconds', 300)  # 5 minutes
        self._cache_max_size = config_dict.get('cache_max_size', 1000)

        logger.info("Firebase authentication provider created")
        if credentials_path:
            logger.info(f"Using credentials from: {credentials_path}")
        if project_id:
            logger.info(f"Firebase project ID: {project_id}")

    async def initialize(self) -> None:
        """Initialize the Firebase authentication provider.

        This method can be called during application startup to perform
        any necessary initialization that might take time or fail.
        """
        if self._initialized:
            return

        try:
            # Perform any async initialization here
            await asyncio.sleep(0.1)  # Placeholder for actual initialization

            self._initialized = True
            logger.info("Firebase authentication provider initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Firebase auth provider: {e}")
            raise

    async def verify_token(self, token: str) -> User | None:
        """Verify a Firebase ID token and return user information.

        Args:
            token: Firebase ID token to verify

        Returns:
            User object if token is valid, None otherwise
        """
        if not self._initialized:
            logger.warning(
                "Auth provider not initialized, attempting lazy initialization"
            )
            await self.initialize()

        try:
            # Use the existing Firebase auth verification
            from clarity.auth.firebase_auth import auth

            decoded_token = auth.verify_id_token(token)

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
            logger.debug(f"Token verification failed: {e}")
            return None

    async def create_user(self, user_data: dict[str, Any]) -> User:
        """Create a new user account.

        Args:
            user_data: User information for account creation

        Returns:
            Created user object
        """
        if not self._initialized:
            await self.initialize()

        try:
            from clarity.auth.firebase_auth import auth

            # Create user in Firebase
            user_record = auth.create_user(
                email=user_data.get("email"),
                display_name=user_data.get("display_name"),
                email_verified=False,
            )

            user = User(
                uid=user_record.uid,
                email=user_record.email,
                display_name=user_record.display_name,
                email_verified=user_record.email_verified,
                firebase_token=None,
                created_at=None,
                last_login=None,
                profile=None,
            )

            logger.info(f"Created new user: {user.uid}")
            return user

        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise

    async def get_user_info(self, user_id: str) -> User | None:
        """Get user information by user ID.

        Args:
            user_id: Firebase user ID

        Returns:
            User object if found, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        try:
            from clarity.auth.firebase_auth import auth

            user_record = auth.get_user(user_id)

            user = User(
                uid=user_record.uid,
                email=user_record.email,
                display_name=user_record.display_name,
                email_verified=user_record.email_verified,
                firebase_token=None,
                created_at=None,
                last_login=None,
                profile=None,
            )

            return user

        except Exception as e:
            logger.debug(f"Failed to get user info: {e}")
            return None

    async def cleanup(self) -> None:
        """Cleanup resources when shutting down."""
        self._token_cache.clear()
        logger.info("Firebase authentication provider cleanup complete")
        self._initialized = False
