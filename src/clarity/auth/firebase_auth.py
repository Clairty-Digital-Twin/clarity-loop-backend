"""CLARITY Digital Twin Platform - Firebase Authentication.

Enterprise-grade Firebase authentication middleware with:
- JWT token validation and verification
- Role-based access control (RBAC)
- Token caching for performance optimization
- Comprehensive audit logging for HIPAA compliance
- User context management
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from functools import wraps
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import FastAPI
    from starlette.responses import Response

from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer
import firebase_admin  # type: ignore[import-untyped]
from firebase_admin import auth, credentials  # type: ignore[import-untyped]
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from clarity.core.interfaces import IAuthProvider
from clarity.models.auth import AuthErrorException as AuthError
from clarity.models.auth import Permission, UserContext, UserRole

if TYPE_CHECKING:
    from clarity.core.config import MiddlewareConfig

# Configure logger
logger = logging.getLogger(__name__)


class FirebaseAuthProvider(IAuthProvider):
    """Firebase authentication provider implementing IAuthProvider interface.

    Following Clean Architecture and SOLID principles:
    - Single Responsibility: Only handles Firebase authentication
    - Open/Closed: Can be extended without modification
    - Liskov Substitution: Can substitute any IAuthProvider  # cSpell:ignore Liskov
    - Interface Segregation: Implements only needed methods
    - Dependency Inversion: Depends on abstractions
    """

    def __init__(
        self,
        credentials_path: str | None = None,
        project_id: str | None = None,
        middleware_config: MiddlewareConfig | None = None,
    ) -> None:
        """Initialize Firebase authentication provider.

        Args:
            credentials_path: Path to Firebase service account credentials
            project_id: Firebase project ID
            middleware_config: Middleware configuration object
        """
        self.project_id = project_id
        self.credentials_path = credentials_path
        self._initialized = False
        self._middleware_config = middleware_config

        # Token cache for performance optimization
        self._token_cache: dict[str, dict[str, Any]] = {}
        self._cache_lock = asyncio.Lock()

        # Configure cache settings from middleware config
        if middleware_config:
            self._cache_enabled = middleware_config.cache_enabled
            self._cache_ttl = middleware_config.cache_ttl_seconds
            self._cache_max_size = middleware_config.cache_max_size
            self._initialization_timeout = (
                middleware_config.initialization_timeout_seconds
            )
        else:
            # Default settings
            self._cache_enabled = True
            self._cache_ttl = 300  # 5 minutes
            self._cache_max_size = 1000
            self._initialization_timeout = 8

    async def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify Firebase authentication token.

        Args:
            token: Firebase ID token to verify

        Returns:
            User information if token is valid, None otherwise
        """
        try:
            # Initialize Firebase if not already done
            await self._ensure_initialized()

            # Check cache first (only if caching is enabled)
            if self._cache_enabled:
                cached_user = await self._get_cached_user(token)
                if cached_user:
                    return cached_user

            # Verify token with Firebase
            decoded_token = auth.verify_id_token(token)  # type: ignore[misc]

            # Get user record for additional info
            user_record = auth.get_user(decoded_token["uid"])  # type: ignore[misc]

            # Create user info dict
            user_info: dict[str, Any] = {
                "user_id": decoded_token["uid"],
                "email": user_record.email,  # type: ignore[misc]
                "name": user_record.display_name,  # type: ignore[misc]
                "verified": user_record.email_verified,  # type: ignore[misc]
                "roles": self._extract_roles(decoded_token),  # type: ignore[arg-type]
                "custom_claims": decoded_token.get("custom_claims", {}),  # type: ignore[misc]
                "created_at": datetime.fromtimestamp(
                    user_record.user_metadata.creation_timestamp / 1000, tz=UTC  # type: ignore[misc,arg-type]
                ).isoformat(),
                "last_login": (
                    datetime.fromtimestamp(
                        user_record.user_metadata.last_sign_in_timestamp / 1000, tz=UTC  # type: ignore[misc,arg-type]
                    ).isoformat()
                    if user_record.user_metadata.last_sign_in_timestamp  # type: ignore[misc]
                    else None
                ),
            }

            # Cache the result (only if caching is enabled)
            if self._cache_enabled:
                await self._cache_user(token, user_info)

        except auth.ExpiredIdTokenError:
            logger.warning("Firebase token expired")
            return None
        except auth.RevokedIdTokenError:
            logger.warning("Firebase token revoked")
            return None
        except auth.InvalidIdTokenError:
            logger.warning("Invalid Firebase token")
            return None
        except Exception:
            logger.exception("Firebase token verification failed")
            return None
        else:
            return user_info

    async def get_user_info(self, user_id: str) -> dict[str, Any] | None:
        """Get user information by ID from Firebase.

        Args:
            user_id: Firebase user ID

        Returns:
            User information if found, None otherwise
        """
        try:
            # Initialize Firebase if not already done
            await self._ensure_initialized()

            # Get user record from Firebase
            user_record = auth.get_user(user_id)  # type: ignore[misc]

            return {
                "user_id": user_record.uid,  # type: ignore[misc]
                "email": user_record.email,  # type: ignore[misc]
                "name": user_record.display_name,  # type: ignore[misc]
                "verified": user_record.email_verified,  # type: ignore[misc]
                "disabled": user_record.disabled,  # type: ignore[misc]
                "created_at": datetime.fromtimestamp(
                    user_record.user_metadata.creation_timestamp / 1000, tz=UTC  # type: ignore[misc,arg-type]
                ).isoformat(),
                "last_login": (
                    datetime.fromtimestamp(
                        user_record.user_metadata.last_sign_in_timestamp / 1000, tz=UTC  # type: ignore[misc,arg-type]
                    ).isoformat()
                    if user_record.user_metadata.last_sign_in_timestamp  # type: ignore[misc]
                    else None
                ),
            }

        except auth.UserNotFoundError:
            logger.warning("Firebase user not found: %s", user_id)
            return None
        except Exception:
            logger.exception("Failed to get Firebase user info")
            return None

    async def initialize(self) -> None:
        """Initialize Firebase Admin SDK."""
        await self._ensure_initialized()

    async def cleanup(self) -> None:
        """Clean up Firebase resources."""
        # Clear token cache
        async with self._cache_lock:
            self._token_cache.clear()

    async def _ensure_initialized(self) -> None:
        """Ensure Firebase Admin SDK is initialized."""
        if self._initialized:
            return

        def _raise_missing_config() -> None:
            """Abstract raise to inner function."""
            msg = "Either credentials_path or project_id must be provided"
            raise ValueError(msg)

        try:
            # Check if Firebase is already initialized
            try:
                firebase_admin.get_app()  # type: ignore[misc]
            except ValueError:
                # No app exists, need to initialize
                pass
            else:
                self._initialized = True
                return

            if self.credentials_path:
                cred = credentials.Certificate(self.credentials_path)  # type: ignore[misc]
                firebase_admin.initialize_app(cred, {"projectId": self.project_id})  # type: ignore[misc]
                logger.info("Firebase Admin SDK initialized with credentials")
            elif self.project_id:
                # Use default credentials (useful for deployed environments)
                cred = credentials.ApplicationDefault()  # type: ignore[misc]
                firebase_admin.initialize_app(cred, {"projectId": self.project_id})  # type: ignore[misc]
                logger.info("Firebase Admin SDK initialized with default credentials")
            else:
                _raise_missing_config()

            self._initialized = True
            logger.info("✅ Firebase Admin SDK initialized successfully")

        except Exception:
            logger.exception("💥 Failed to initialize Firebase Admin SDK")
            # Continue without Firebase - graceful degradation
            # In production, you might want to raise the exception

    @staticmethod
    def _extract_roles(decoded_token: dict[str, Any]) -> list[str]:
        """Extract user roles from Firebase token."""
        # Extract role from custom claims
        custom_claims = decoded_token.get("custom_claims", {})
        role = custom_claims.get("role", "patient")

        # Return as list for consistency
        return [role]

    async def _get_cached_user(self, token: str) -> dict[str, Any] | None:
        """Get user info from cache if valid."""
        async with self._cache_lock:
            cache_entry = self._token_cache.get(token)

            if not cache_entry:
                return None

            # Check if cache entry is still valid
            if time.time() - cache_entry["timestamp"] > self._cache_ttl:
                del self._token_cache[token]
                return None

            user_info: dict[str, Any] = cache_entry["user_info"]
            return user_info

    async def _cache_user(self, token: str, user_info: dict[str, Any]) -> None:
        """Cache user info for token with size limits and TTL management."""
        if not self._cache_enabled:
            return

        async with self._cache_lock:
            # Clean up expired entries first
            await self._cleanup_expired_cache_entries()

            # If cache is at max size, remove oldest entries
            if len(self._token_cache) >= self._cache_max_size:
                # Remove oldest entries (FIFO)
                oldest_tokens = sorted(
                    self._token_cache.items(), key=lambda x: x[1]["timestamp"]
                )[: len(self._token_cache) - self._cache_max_size + 1]

                for token_to_remove, _ in oldest_tokens:
                    del self._token_cache[token_to_remove]

            # Add new entry
            self._token_cache[token] = {
                "user_info": user_info,
                "timestamp": time.time(),
            }

    async def _cleanup_expired_cache_entries(self) -> None:
        """Remove expired cache entries based on TTL."""
        if not self._cache_enabled:
            return

        current_time = time.time()
        expired_tokens = [
            token
            for token, entry in self._token_cache.items()
            if current_time - entry["timestamp"] > self._cache_ttl
        ]

        for expired_token in expired_tokens:
            del self._token_cache[expired_token]

        if expired_tokens:
            logger.debug("Cleaned up %d expired cache entries", len(expired_tokens))


class FirebaseAuthMiddleware(BaseHTTPMiddleware):
    """Firebase authentication middleware for FastAPI.

    Features:
    - JWT token validation using Firebase Admin SDK
    - Token caching for improved performance
    - Role-based access control
    - Comprehensive error handling and logging
    - HIPAA-compliant audit trails
    """

    def __init__(
        self,
        app: FastAPI,
        auth_provider: IAuthProvider,
        exempt_paths: list[str] | None = None,
    ) -> None:
        """Initialize Firebase authentication middleware.

        Args:
            app: FastAPI application instance
            auth_provider: Authentication provider (dependency injection)
            exempt_paths: Paths that don't require authentication
        """
        super().__init__(app)

        self.auth_provider = auth_provider
        self.exempt_paths = exempt_paths or [
            "/",
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
        ]

        logger.info("Firebase authentication middleware initialized")

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process authentication for incoming requests."""
        # Check if path is exempt from authentication
        if self._is_exempt_path(request.url.path):
            return await call_next(request)

        try:
            # Extract and verify token
            user_context = await self._authenticate_request(request)

            # Attach user context to request state
            request.state.user = user_context

            # Log successful authentication
            logger.info(
                "Authenticated user: %s for %s", user_context.user_id, request.url.path
            )

            return await call_next(request)

        except AuthError as e:
            logger.warning(
                "Authentication failed for %s: %s", request.url.path, e.message
            )
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": e.error_code,
                    "message": e.message,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
        except Exception:
            logger.exception("Token verification error")
            msg = "Token verification failed"
            raise AuthError(msg, 500, "verification_failed") from None

    def _is_exempt_path(self, path: str) -> bool:
        """Check if the path is exempt from authentication."""
        return any(path.startswith(exempt_path) for exempt_path in self.exempt_paths)

    async def _authenticate_request(self, request: Request) -> UserContext:
        """Authenticate request and return user context."""
        # Extract token from Authorization header
        token = self._extract_token(request)

        # Verify token using auth provider
        user_info = await self.auth_provider.verify_token(token)

        if not user_info:
            msg = "Token verification failed"
            raise AuthError(msg, 401, "invalid_token")

        # Create user context from verified user info
        return self._create_user_context(user_info)

    @staticmethod
    def _extract_token(request: Request) -> str:
        """Extract Bearer token from Authorization header."""
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            msg = "Missing Authorization header"
            raise AuthError(msg, 401, "missing_token")

        if not auth_header.startswith("Bearer "):
            msg = "Invalid Authorization header format"
            raise AuthError(msg, 401, "invalid_token_format")

        token = auth_header[7:]  # Remove "Bearer " prefix

        if not token:
            msg = "Empty authentication token"
            raise AuthError(msg, 401, "empty_token")

        return token

    @staticmethod
    def _create_user_context(user_info: dict[str, Any]) -> UserContext:
        """Create user context from verified user info."""
        try:
            # Extract role from user info
            roles = user_info.get("roles", ["patient"])
            role_str = roles[0] if roles else "patient"

            try:
                role = UserRole(role_str)
            except ValueError:
                logger.warning(
                    "Invalid role '%s' for user %s, defaulting to patient",
                    role_str,
                    user_info.get("user_id"),
                )
                role = UserRole.PATIENT

            # Role-permission mapping
            role_permissions = {
                UserRole.PATIENT: [Permission.READ_OWN_DATA, Permission.WRITE_OWN_DATA],
                UserRole.CLINICIAN: [
                    Permission.READ_OWN_DATA,
                    Permission.WRITE_OWN_DATA,
                    Permission.READ_PATIENT_DATA,
                    Permission.WRITE_PATIENT_DATA,
                ],
                UserRole.RESEARCHER: [Permission.READ_ANONYMIZED_DATA],
                UserRole.ADMIN: [
                    Permission.READ_OWN_DATA,
                    Permission.WRITE_OWN_DATA,
                    Permission.READ_PATIENT_DATA,
                    Permission.WRITE_PATIENT_DATA,
                    Permission.READ_ANONYMIZED_DATA,
                    Permission.MANAGE_USERS,
                    Permission.SYSTEM_ADMIN,
                ],
            }

            # Get permissions for the role
            permissions = role_permissions.get(role, [])

            return UserContext(
                user_id=user_info["user_id"],
                email=user_info.get("email"),
                role=role,
                permissions=permissions,
                is_verified=user_info.get("verified", False),
                is_active=not user_info.get("disabled"),
                custom_claims=user_info.get("custom_claims", {}),
                created_at=(
                    datetime.fromisoformat(user_info["created_at"])
                    if user_info.get("created_at")
                    else datetime.now(UTC)
                ),
                last_login=(
                    datetime.fromisoformat(user_info["last_login"])
                    if user_info.get("last_login")
                    else None
                ),
            )

        except Exception:
            logger.exception("Error creating user context")
            msg = "Failed to create user context"
            raise AuthError(msg, 500, "context_creation_failed") from None


# Dependency injection functions for FastAPI

security = HTTPBearer()


def get_current_user(request: Request) -> UserContext:
    """FastAPI dependency to get current authenticated user."""
    if not hasattr(request.state, "user"):
        msg = "User not authenticated. Authentication middleware not applied?"
        raise HTTPException(status_code=401, detail=msg)
    user: UserContext = request.state.user
    return user


def require_auth(
    permissions: list[Permission] | None = None, roles: list[UserRole] | None = None
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Decorator to require authentication and specific permissions/roles.

    Args:
        permissions: Required permissions
        roles: Required roles

    Returns:
        Decorated function that enforces authentication
    """

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args: object, **kwargs: object) -> object:
            def _raise_missing_request() -> None:
                """Abstract raise to inner function."""
                msg = "Request object not found"
                raise HTTPException(status_code=500, detail=msg)

            def _raise_role_unauthorized(role: str) -> None:
                """Abstract raise to inner function."""
                msg = f"Role '{role}' not authorized"
                raise HTTPException(status_code=403, detail=msg)

            def _raise_missing_permissions(missing: list[Permission]) -> None:
                """Abstract raise to inner function."""
                msg = f"Missing required permissions: {list(missing)}"
                raise HTTPException(status_code=403, detail=msg)

            def _raise_disabled_account() -> None:
                """Abstract raise to inner function."""
                raise HTTPException(status_code=403, detail="User account is disabled")

            # Extract request from args/kwargs
            request: Request | None = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                _raise_missing_request()

            # Get user context (not await since get_current_user is not async)
            # We know request is not None here due to check above
            user_context = get_current_user(request)  # type: ignore[arg-type]

            # Check role requirements
            if roles and user_context.role not in roles:
                _raise_role_unauthorized(str(user_context.role))

            # Check permission requirements
            if permissions:
                missing_permissions = set(permissions) - set(user_context.permissions)
                if missing_permissions:
                    _raise_missing_permissions(list(missing_permissions))

            # Check if user is active
            if not user_context.is_active:
                _raise_disabled_account()

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_role(
    *roles: UserRole,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Convenience decorator to require specific roles."""
    return require_auth(roles=list(roles))


def require_permission(
    *permissions: Permission,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Convenience decorator to require specific permissions."""
    return require_auth(permissions=list(permissions))
