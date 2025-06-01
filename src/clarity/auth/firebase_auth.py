"""CLARITY Digital Twin Platform - Firebase Authentication

Enterprise-grade Firebase authentication middleware with:
- JWT token validation and verification
- Role-based access control (RBAC)
- Token caching for performance optimization
- Comprehensive audit logging for HIPAA compliance
- User context management
"""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
import logging
import time
from typing import Any

from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer
import firebase_admin
from firebase_admin import auth, credentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from .models import AuthError, Permission, TokenInfo, UserContext, UserRole

# Configure logger
logger = logging.getLogger(__name__)


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
        app: Any,
        credentials_path: str | None = None,
        project_id: str | None = None,
        exempt_paths: list[str] | None = None,
        cache_ttl: int = 300,  # 5 minutes
        enable_caching: bool = True,
    ) -> None:
        """Initialize Firebase authentication middleware.

        Args:
            app: FastAPI application instance
            credentials_path: Path to Firebase service account credentials
            project_id: Firebase project ID
            exempt_paths: Paths that don't require authentication
            cache_ttl: Token cache time-to-live in seconds
            enable_caching: Enable token caching for performance
        """
        super().__init__(app)

        self.project_id = project_id
        self.exempt_paths = exempt_paths or [
            "/",
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
        ]
        self.cache_ttl = cache_ttl
        self.enable_caching = enable_caching

        # Token cache for performance optimization
        self._token_cache: dict[str, dict[str, Any]] = {}
        self._cache_lock = asyncio.Lock()

        # Initialize Firebase Admin SDK
        self._init_firebase(credentials_path, project_id)

        # Role-permission mapping
        self._role_permissions = {
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

        logger.info("Firebase authentication middleware initialized")

    def _init_firebase(
        self, credentials_path: str | None, project_id: str | None
    ) -> None:
        """Initialize Firebase Admin SDK."""
        try:
            if not firebase_admin._apps:
                if credentials_path:
                    cred = credentials.Certificate(credentials_path)
                    firebase_admin.initialize_app(cred, {"projectId": project_id})
                else:
                    # Use default credentials (ADC)
                    firebase_admin.initialize_app()
                logger.info("Firebase Admin SDK initialized for authentication")
            else:
                logger.info("Firebase Admin SDK already initialized")
        except Exception as e:
            logger.exception("Failed to initialize Firebase Admin SDK")
            msg = f"Firebase initialization failed: {e}"
            raise AuthError(msg, status_code=500) from None

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Any]
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

            response = await call_next(request)
            return response

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

        # Check cache first
        if self.enable_caching:
            cached_user = await self._get_cached_user(token)
            if cached_user:
                return cached_user

        # Verify token with Firebase
        token_info = await self._verify_firebase_token(token)

        # Create user context
        user_context = await self._create_user_context(token_info)

        # Cache the result
        if self.enable_caching:
            await self._cache_user(token, user_context)

        return user_context

    def _extract_token(self, request: Request) -> str:
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

    async def _verify_firebase_token(self, token: str) -> TokenInfo:
        """Verify Firebase ID token."""
        try:
            # Verify the token with Firebase Admin SDK
            decoded_token = auth.verify_id_token(token)

            token_info = TokenInfo(
                token=token,
                user_id=decoded_token["uid"],
                email=decoded_token.get("email"),
                issued_at=datetime.fromtimestamp(decoded_token["iat"], tz=UTC),
                expires_at=datetime.fromtimestamp(decoded_token["exp"], tz=UTC),
                is_admin=decoded_token.get("admin", False),
                custom_claims=decoded_token.get("custom_claims", {}),
            )

            # Check if token is expired
            if token_info.expires_at < datetime.now(UTC):
                msg = "Token has expired"
                raise AuthError(msg, 401, "token_expired") from None

            return token_info

        except auth.ExpiredIdTokenError:
            msg = "Authentication token has expired"
            raise AuthError(msg, 401, "token_expired") from None
        except auth.RevokedIdTokenError:
            msg = "Authentication token has been revoked"
            raise AuthError(msg, 401, "token_revoked") from None
        except auth.InvalidIdTokenError:
            msg = "Invalid authentication token"
            raise AuthError(msg, 401, "invalid_token") from None
        except auth.CertificateFetchError:
            msg = "Unable to verify token"
            raise AuthError(msg, 500, "verification_error") from None
        except Exception:
            logger.exception("Token verification error")
            msg = "Token verification failed"
            raise AuthError(msg, 500, "verification_failed") from None

    async def _create_user_context(self, token_info: TokenInfo) -> UserContext:
        """Create user context from verified token."""
        try:
            # Get user record from Firebase
            user_record = auth.get_user(token_info.user_id)

            # Extract role from custom claims
            role_claim = token_info.custom_claims.get("role", "patient")
            try:
                role = UserRole(role_claim)
            except ValueError:
                logger.warning(
                    "Invalid role '%s' for user %s, defaulting to patient",
                    role_claim,
                    token_info.user_id,
                )
                role = UserRole.PATIENT

            # Get permissions for the role
            permissions = self._role_permissions.get(role, [])

            user_context = UserContext(
                user_id=token_info.user_id,
                email=user_record.email,
                role=role,
                permissions=permissions,
                is_verified=user_record.email_verified,
                is_active=not user_record.disabled,
                custom_claims=token_info.custom_claims,
                created_at=datetime.fromtimestamp(
                    user_record.user_metadata.creation_timestamp / 1000, tz=UTC
                ),
                last_login=(
                    datetime.fromtimestamp(
                        user_record.user_metadata.last_sign_in_timestamp / 1000, tz=UTC
                    )
                    if user_record.user_metadata.last_sign_in_timestamp
                    else None
                ),
            )

            return user_context

        except auth.UserNotFoundError:
            msg = "User not found"
            raise AuthError(msg, 401, "user_not_found") from None
        except Exception as e:
            logger.exception("Error creating user context")
            msg = "Failed to create user context"
            raise AuthError(msg, 500, "context_creation_failed") from None

    async def _get_cached_user(self, token: str) -> UserContext | None:
        """Get user context from cache if valid."""
        async with self._cache_lock:
            cache_entry = self._token_cache.get(token)

            if not cache_entry:
                return None

            # Check if cache entry is still valid
            if time.time() - cache_entry["timestamp"] > self.cache_ttl:
                del self._token_cache[token]
                return None

            return cache_entry["user_context"]

    async def _cache_user(self, token: str, user_context: UserContext) -> None:
        """Cache user context for token."""
        async with self._cache_lock:
            self._token_cache[token] = {
                "user_context": user_context,
                "timestamp": time.time(),
            }

            # Clean up expired cache entries (simple cleanup)
            current_time = time.time()
            expired_tokens = [
                t
                for t, entry in self._token_cache.items()
                if current_time - entry["timestamp"] > self.cache_ttl
            ]

            for expired_token in expired_tokens:
                del self._token_cache[expired_token]


# Dependency injection functions for FastAPI

security = HTTPBearer()


async def get_current_user(request: Request) -> UserContext:
    """FastAPI dependency to get current authenticated user."""
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="Authentication required")

    return request.state.user


def require_auth(
    permissions: list[Permission] | None = None, roles: list[UserRole] | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to require authentication and specific permissions/roles.

    Args:
        permissions: Required permissions
        roles: Required roles

    Returns:
        Decorated function that enforces authentication
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                msg = "Request object not found"
                raise HTTPException(status_code=500, detail=msg)

            # Get user context
            user_context = await get_current_user(request)

            # Check role requirements
            if roles and user_context.role not in roles:
                msg = f"Role '{user_context.role}' not authorized"
                raise HTTPException(status_code=403, detail=msg)

            # Check permission requirements
            if permissions:
                missing_permissions = set(permissions) - set(user_context.permissions)
                if missing_permissions:
                    msg = f"Missing required permissions: {list(missing_permissions)}"
                    raise HTTPException(status_code=403, detail=msg)

            # Check if user is active
            if not user_context.is_active:
                raise HTTPException(status_code=403, detail="User account is disabled")

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_role(
    *roles: UserRole,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Convenience decorator to require specific roles."""
    return require_auth(roles=list(roles))


def require_permission(
    *permissions: Permission,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Convenience decorator to require specific permissions."""
    return require_auth(permissions=list(permissions))
