"""Debug endpoints for testing authentication.

THIS SHOULD BE REMOVED OR DISABLED IN PRODUCTION!
"""

from datetime import UTC, datetime
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from clarity.auth.dependencies import AuthenticatedUser
from clarity.models.auth import UserContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/token-info")
async def debug_token_info(
    request: Request,
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    """Debug endpoint to check token parsing and middleware state."""
    logger.info("🔍 Debug token info requested")

    # Check if authorization header exists
    if not authorization:
        return {
            "error": "No Authorization header",
            "headers": dict(request.headers),
        }

    # Parse Bearer token
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return {
            "error": "Invalid Authorization format",
            "authorization": authorization,
            "expected": "Bearer <token>",
        }

    token = parts[1]

    # Basic token info
    token_parts = token.split(".")

    return {
        "token_format": "Valid JWT" if len(token_parts) == 3 else "Invalid JWT format",
        "token_length": len(token),
        "token_preview": f"{token[:20]}...{token[-20:]}",
        "header_parts": len(token_parts),
        "request_state": {
            "has_user": hasattr(request.state, "user"),
            "user": str(request.state.user) if hasattr(request.state, "user") else None,
        },
    }


@router.get("/auth-check")
async def debug_auth_check(
    current_user: AuthenticatedUser,
) -> dict[str, Any]:
    """Debug endpoint that requires authentication."""
    logger.info("✅ Auth check passed for user: %s", current_user.user_id)

    return {
        "authenticated": True,
        "user_id": current_user.user_id,
        "email": current_user.email,
        "display_name": current_user.custom_claims.get("display_name", "N/A"),
        "email_verified": current_user.is_verified,
        "token_exp": getattr(current_user, "token_exp", "N/A"),
    }


@router.get("/echo-headers")
async def debug_echo_headers(request: Request) -> dict[str, Any]:
    """Echo all headers for debugging."""
    return {
        "headers": dict(request.headers),
        "url": str(request.url),
        "method": request.method,
    }


@router.get("/test-middleware")
async def debug_test_middleware(request: Request) -> dict[str, Any]:
    """Test endpoint that should trigger middleware (not in exempt paths)."""
    logger.warning("🔥🔥 TEST-MIDDLEWARE ENDPOINT HIT")

    # This endpoint is NOT in the exempt paths, so middleware should run
    return {
        "message": "If you see this without auth, middleware is not running",
        "request_state": {
            "has_user": hasattr(request.state, "user"),
            "user": str(request.state.user) if hasattr(request.state, "user") else None,
        },
        "path": request.url.path,
        "headers": {
            "authorization": request.headers.get("authorization", "NOT_PROVIDED")
        },
    }


@router.get("/middleware-stack")
async def debug_middleware_stack(request: Request) -> dict[str, Any]:
    """Debug endpoint to check middleware stack."""
    from starlette.middleware import Middleware

    # Try to access the app instance
    app = request.app

    # Get middleware information
    middleware_info = []

    # Check if middleware stack is accessible
    if hasattr(app, "middleware_stack"):
        middleware_info.append({"middleware_stack": str(app.middleware_stack)})

    # Check if user_middleware is accessible
    if hasattr(app, "user_middleware"):
        for idx, mw in enumerate(app.user_middleware):
            middleware_info.append({f"user_middleware_{idx}": str(mw)})

    # Check built middleware
    if hasattr(app, "middleware") and hasattr(app.middleware, "cls"):
        middleware_info.append({"built_middleware_class": str(app.middleware.cls)})

    # Check if middleware stack was built
    middleware_info.append({"middleware_stack_built": hasattr(app, "middleware_stack")})

    # Check request state
    state_info = {}
    if hasattr(request, "state"):
        state_info = {
            "has_user": hasattr(request.state, "user"),
            "user": str(request.state.user) if hasattr(request.state, "user") else None,
        }

    # Check app instance ID
    app_id = id(app)

    return {
        "app_instance_id": app_id,
        "app_title": getattr(app, "title", "Unknown"),
        "middleware_info": middleware_info,
        "request_state": state_info,
        "app_attributes": [attr for attr in dir(app) if "middleware" in attr.lower()],
    }


@router.post("/verify-token-directly")
async def debug_verify_token_directly(
    request: Request,
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    """Debug endpoint to test token verification directly."""
    logger.warning("🔍🔍 DEBUG VERIFY TOKEN DIRECTLY CALLED")

    # Get the container to access auth provider
    from clarity.core.container import get_container

    container = get_container()
    auth_provider = container.get_auth_provider()

    result = {
        "timestamp": datetime.now(UTC).isoformat(),
        "auth_provider_type": type(auth_provider).__name__,
        "auth_provider_initialized": getattr(auth_provider, "_initialized", False),
    }

    # Check authorization header
    if not authorization:
        result["error"] = "No Authorization header provided"
        return result

    # Parse token
    if not authorization.startswith("Bearer "):
        result["error"] = "Invalid Authorization format (expected 'Bearer <token>')"
        return result

    token = authorization[7:]
    result["token_length"] = len(token)
    result["token_preview"] = f"{token[:20]}...{token[-20:]}"

    # Try to verify token directly
    try:
        logger.warning("🔍🔍 Calling auth_provider.verify_token() directly")
        user_info = await auth_provider.verify_token(token)

        if user_info:
            result["verification_success"] = True
            result["user_info"] = {
                "user_id": user_info.get("user_id"),
                "email": user_info.get("email"),
                "verified": user_info.get("verified"),
                "roles": user_info.get("roles", []),
            }
            logger.warning("✅✅ Direct token verification SUCCEEDED")
        else:
            result["verification_success"] = False
            result["user_info"] = None
            result["note"] = (
                "verify_token returned None - check logs for authentication error"
            )
            logger.warning("❌❌ Direct token verification FAILED - returned None")

    except Exception as e:
        result["verification_success"] = False
        result["exception_type"] = type(e).__name__
        result["exception_message"] = str(e)
        logger.exception("❌❌ Direct token verification threw exception")

    # Check AWS Cognito status
    try:
        import boto3
        cognito_client = boto3.client('cognito-idp', region_name=os.getenv("AWS_REGION", "us-east-1"))
        result["cognito_initialized"] = True
        result["cognito_user_pool_id"] = os.getenv("COGNITO_USER_POOL_ID", "NOT_SET")
        result["cognito_region"] = os.getenv("COGNITO_REGION", "us-east-1")
    except Exception as e:
        result["cognito_initialized"] = False
        result["cognito_note"] = f"Cognito client error: {str(e)}"

    # Check environment

    result["environment"] = {
        "AWS_REGION": os.environ.get("AWS_REGION", "NOT_SET"),
        "COGNITO_USER_POOL_ID": os.environ.get("COGNITO_USER_POOL_ID", "NOT_SET"),
        "ENVIRONMENT": os.environ.get("ENVIRONMENT", "NOT_SET"),
    }

    return result
