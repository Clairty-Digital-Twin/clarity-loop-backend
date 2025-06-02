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

from clarity.auth.models import AuthError, Permission, UserContext, UserRole
from clarity.core.interfaces import IAuthProvider

# Configure logger
logger = logging.getLogger(__name__)


class FirebaseAuthProvider(IAuthProvider):
    """Firebase authentication provider implementing IAuthProvider interface.

    Following Clean Architecture and SOLID principles:
    - Single Responsibility: Only handles Firebase authentication
    - Open/Closed: Can be extended without modification
    - Liskov Substitution: Can substitute any IAuthProvider
    - Interface Segregation: Implements only needed methods
    - Dependency Inversion: Depends on abstractions
    """

    def __init__(
        self, credentials_path: str | None = None, project_id: str | None = None
    ) -> None:
        """Initialize Firebase authentication provider.

        Args:
            credentials_path: Path to Firebase service account credentials
            project_id: Firebase project ID
        """
        self.project_id = project_id
        self.credentials_path = credentials_path
        self._initialized = False

        # Token cache for performance optimization
        self._token_cache: dict[str, dict[str, Any]] = {}
        self._cache_lock = asyncio.Lock()
        self._cache_ttl = 300  # 5 minutes

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

            # Check cache first
            cached_user = await self._get_cached_user(token)
            if cached_user:
                return cached_user

            # Verify token with Firebase
            decoded_token = auth.verify_id_token(token)  # type: ignore[misc]

            # Get user record for additional info
            user_record = auth.get_user(decoded_token["uid"])  # type: ignore[misc]

            # Create user info dict
            user_info: dict[str, Any] = {  # type: ignore[misc]
                "user_id": decoded_token["uid"],  # type: ignore[misc]
                "email": user_record.email,  # type: ignore[misc]
                "name": user_record.display_name,  # type: ignore[misc]
                "verified": user_record.email_verified,  # type: ignore[misc]
                "roles": self._extract_roles(decoded_token),  # type: ignore[misc]
                "custom_claims": decoded_token.get("custom_claims", {}),  # type: ignore[misc]
                "created_at": datetime.fromtimestamp(
                    user_record.user_metadata.creation_timestamp / 1000, tz=UTC  # type: ignore[misc]
                ).isoformat(),
                "last_login": (
                    datetime.fromtimestamp(
                        user_record.user_metadata.last_sign_in_timestamp / 1000, tz=UTC  # type: ignore[misc]
                    ).isoformat()
                    if user_record.user_metadata.last_sign_in_timestamp  # type: ignore[misc]
                    else None
                ),
            }

            # Cache the result
            await self._cache_user(token, user_info)  # type: ignore[misc]

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
            return user_info  # type: ignore[misc]

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
                    user_record.user_metadata.creation_timestamp / 1000, tz=UTC  # type: ignore[misc]
                ).isoformat(),
                "last_login": (
                    datetime.fromtimestamp(
                        user_record.user_metadata.last_sign_in_timestamp / 1000, tz=UTC  # type: ignore[misc]
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
                firebase_admin.get_app()
            except ValueError:
                # No app exists, need to initialize
                pass
            else:
                self._initialized = True
                return

            if self.credentials_path:
                cred = credentials.Certificate(self.credentials_path)
                firebase_admin.initialize_app(cred, {"projectId": self.project_id})  # type: ignore[misc]
                logger.info("Firebase Admin SDK initialized with credentials")
            elif self.project_id:
                # Use default credentials (useful for deployed environments)
                cred = credentials.ApplicationDefault()
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
        """Cache user info for token."""
        async with self._cache_lock:
            self._token_cache[token] = {
                "user_info": user_info,
                "timestamp": time.time(),
            }

            # Clean up expired cache entries
            current_time = time.time()
            expired_tokens = [
                t
                for t, entry in self._token_cache.items()
                if current_time - entry["timestamp"] > self._cache_ttl
            ]

            for expired_token in expired_tokens:
                del self._token_cache[expired_token]


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
