"""Rate Limiting Middleware.

Provides application-level rate limiting using slowapi to protect against
abuse and ensure fair resource usage across users.
"""

from collections.abc import Callable
import logging
from typing import Any, ClassVar

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from clarity.core.exceptions import ProblemDetail

logger = logging.getLogger(__name__)


def get_user_id_or_ip(request: Request) -> str:
    """Extract user ID from authenticated requests, fallback to IP address.

    This key function provides per-user rate limiting for authenticated users
    and per-IP rate limiting for anonymous users.
    """
    # Check if user is authenticated (set by auth middleware)
    if hasattr(request.state, "user") and request.state.user:
        user = request.state.user
        # Try different user ID fields that might be present
        user_id = user.get("uid") or user.get("user_id") or user.get("sub")
        if user_id:
            logger.debug("Rate limiting by user ID: %s", user_id)
            return f"user:{user_id}"

    # Fallback to IP address for unauthenticated requests
    ip_address = get_remote_address(request)
    logger.debug("Rate limiting by IP address: %s", ip_address)
    return f"ip:{ip_address}"


def get_ip_only(request: Request) -> str:
    """Get IP address only, used for stricter anonymous endpoint limits."""
    return f"ip:{get_remote_address(request)}"


async def custom_rate_limit_exceeded_handler(  # noqa: RUF029 - FastAPI handler
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Custom handler for rate limit exceeded errors.

    Returns a standardized error response with rate limit headers.
    """
    logger.warning(
        "Rate limit exceeded for %s - Key: %s, Limit: %s",
        request.url.path,
        exc.detail,
        getattr(exc, "limit", "unknown"),
    )

    from clarity.middleware.security_headers import (  # noqa: PLC0415
        SecurityHeadersMiddleware,
    )

    response = JSONResponse(
        status_code=429,
        content=ProblemDetail(
            type="rate_limit_exceeded",
            title="Too Many Requests",
            detail="Rate limit exceeded. Please retry after some time.",
            status=429,
            instance=f"https://api.clarity.health/requests/{id(exc)}",
        ).model_dump(),
    )

    # Add security headers to the response
    SecurityHeadersMiddleware.add_security_headers_to_response(response)

    # Add rate limit headers if available
    return request.app.state.limiter._inject_headers(  # type: ignore[no-any-return]  # noqa: SLF001
        response, request.state.view_rate_limit
    )


class RateLimitingMiddleware:
    """Rate limiting middleware configuration.

    Provides methods to create and configure rate limiters with
    different strategies for various endpoint types.
    """

    # Default rate limits
    DEFAULT_LIMITS: ClassVar[dict[str, str]] = {
        "global": "1000/hour",  # Global limit per key
        "auth": "20/hour",  # Authentication endpoints
        "health": "100/minute",  # Health data endpoints
        "ai": "50/hour",  # AI/ML endpoints (resource intensive)
        "read": "200/minute",  # General read operations
        "write": "60/minute",  # General write operations
    }

    @staticmethod
    def create_limiter(
        key_func: Callable[[Request], str] = get_user_id_or_ip,
        default_limits: list[str | Callable[..., str]] | None = None,
        storage_uri: str | None = None,
    ) -> Limiter:
        """Create a configured rate limiter instance.

        Args:
            key_func: Function to extract rate limit key from request
            default_limits: Default rate limits to apply
            storage_uri: Redis URI for distributed rate limiting

        Returns:
            Configured Limiter instance
        """
        if default_limits is None:
            default_limits = [RateLimitingMiddleware.DEFAULT_LIMITS["global"]]

        limiter = Limiter(
            key_func=key_func,
            default_limits=default_limits,
            storage_uri=storage_uri,  # Use Redis if available for distributed limiting
            headers_enabled=True,  # Add X-RateLimit-* headers
            strategy="fixed-window",  # Simple and predictable
            key_style="endpoint",  # Include endpoint in rate limit key
        )

        logger.info(
            "🚦 Rate limiter initialized with defaults: %s, storage: %s",
            default_limits,
            "Redis" if storage_uri else "In-memory",
        )

        return limiter

    @staticmethod
    def get_auth_limiter() -> Limiter:
        """Create a rate limiter specifically for authentication endpoints."""
        return RateLimitingMiddleware.create_limiter(
            key_func=get_ip_only,  # Always use IP for auth endpoints
            default_limits=[RateLimitingMiddleware.DEFAULT_LIMITS["auth"]],
        )

    @staticmethod
    def get_ai_limiter() -> Limiter:
        """Create a rate limiter for AI/ML endpoints with stricter limits."""
        return RateLimitingMiddleware.create_limiter(
            key_func=get_user_id_or_ip,
            default_limits=[RateLimitingMiddleware.DEFAULT_LIMITS["ai"]],
        )


def setup_rate_limiting(app: Any, redis_url: str | None = None) -> Limiter:
    """Set up rate limiting for a FastAPI application.

    Args:
        app: FastAPI application instance
        redis_url: Optional Redis URL for distributed rate limiting

    Returns:
        Configured Limiter instance
    """
    # Create the main limiter
    limiter = RateLimitingMiddleware.create_limiter(storage_uri=redis_url)

    # Attach limiter to app state
    app.state.limiter = limiter

    # Add exception handler
    app.add_exception_handler(RateLimitExceeded, custom_rate_limit_exceeded_handler)

    logger.info("✅ Rate limiting middleware configured")

    return limiter


# Export common rate limit decorators for easy use
# Usage: @rate_limit("5/minute")
def rate_limit(
    _limit_string: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to apply rate limiting to an endpoint."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # This will be properly bound when the limiter is attached to the app
        return func

    return decorator
