"""Authentication middleware for JWT token validation.

This middleware validates Cognito JWT tokens and populates request.state.user
with the authenticated user context.
"""

# removed - breaks FastAPI

import logging
import os
from typing import TYPE_CHECKING, Any, cast

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from clarity.auth.aws_auth_provider import CognitoAuthProvider
from clarity.auth.modal_auth_fix import set_user_context
from clarity.models.auth import AuthError
from clarity.services.dynamodb_service import DynamoDBService

if TYPE_CHECKING:
    pass  # Only for type stubs now

logger = logging.getLogger(__name__)

# Skip auth for these paths
PUBLIC_PATHS = {
    "/",
    "/health",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/api/v1",
    # NOTE: /api/v1/auth/register removed - now requires ENABLE_SELF_SIGNUP=true
    "/api/v1/auth/login",
    "/api/v1/auth/refresh",
    "/api/v1/auth/health",
    "/api/v1/auth/confirm-email",
    "/api/v1/auth/resend-confirmation",
    "/api/v1/auth/forgot-password",
    "/api/v1/auth/reset-password",
    "/api/v1/test",
}


class CognitoAuthMiddleware(BaseHTTPMiddleware):
    """Middleware to validate Cognito JWT tokens and populate user context."""

    def __init__(self, app: ASGIApp) -> None:
        """Initialize the authentication middleware.

        Args:
            app: The ASGI application
        """
        super().__init__(app)

        # Get configuration from environment
        self.enable_auth = os.getenv("ENABLE_AUTH", "true").lower() == "true"
        self.user_pool_id = os.getenv("COGNITO_USER_POOL_ID", "")
        self.client_id = os.getenv("COGNITO_CLIENT_ID", "")
        self.region = os.getenv("COGNITO_REGION", "us-east-1")

        # Initialize auth provider if enabled
        self.auth_provider: CognitoAuthProvider | None = None
        if self.enable_auth and self.user_pool_id and self.client_id:
            try:
                # Initialize DynamoDB service for user management
                dynamodb_service = DynamoDBService()

                # Initialize Cognito auth provider with DynamoDB
                self.auth_provider = CognitoAuthProvider(
                    user_pool_id=self.user_pool_id,
                    client_id=self.client_id,
                    region=self.region,
                    dynamodb_service=dynamodb_service,
                )
                logger.info("Authentication middleware initialized with Cognito")
            except Exception as e:
                logger.exception("Failed to initialize auth provider: %s", e)
                self.auth_provider = None
        else:
            logger.warning("Authentication disabled or not configured")

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Process the request and validate JWT token if present.

        Args:
            request: The incoming request
            call_next: The next middleware or route handler

        Returns:
            The response from the next handler
        """
        # Initialize request state
        request.state.user = None

        # Skip auth for public paths
        if request.url.path in PUBLIC_PATHS:
            return cast(Response, await call_next(request))

        # Skip auth if disabled
        if not self.enable_auth or not self.auth_provider:
            return cast(Response, await call_next(request))

        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            # No auth header, continue without user context
            return cast(Response, await call_next(request))

        token = auth_header[7:]  # Remove "Bearer " prefix

        try:
            # Initialize auth provider if needed
            if (
                hasattr(self.auth_provider, "is_initialized")
                and not self.auth_provider.is_initialized()
            ):
                await self.auth_provider.initialize()

            # Verify token
            user_info = await self.auth_provider.verify_token(token)
            if user_info:
                # Get or create full user context (includes DynamoDB sync)
                user_context = await self.auth_provider.get_or_create_user_context(
                    user_info
                )

                # Set user context in request state
                request.state.user = user_context

                # Also set in contextvars for Modal compatibility
                set_user_context(user_context)

                logger.debug("User authenticated: %s", user_context.user_id)

        except AuthError as e:
            # Log authentication errors but continue
            # The dependency will handle returning 401
            logger.warning("Authentication failed: %s", e)
        except Exception as e:
            # Log unexpected errors but continue
            logger.exception("Unexpected error during authentication: %s", e)

        # Continue to the next handler
        return cast(Response, await call_next(request))
