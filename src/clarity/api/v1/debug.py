"""Debug endpoints for testing authentication.

THIS SHOULD BE REMOVED OR DISABLED IN PRODUCTION!
"""

import logging
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
        }
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
        "display_name": current_user.custom_claims.get('display_name', 'N/A'),
        "email_verified": current_user.is_verified,
        "firebase_token_exp": getattr(current_user, 'firebase_token_exp', 'N/A'),
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
        }
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
        middleware_info.append({
            "middleware_stack": str(app.middleware_stack)
        })
    
    # Check if user_middleware is accessible
    if hasattr(app, "user_middleware"):
        for idx, mw in enumerate(app.user_middleware):
            middleware_info.append({
                f"user_middleware_{idx}": str(mw)
            })
    
    # Check built middleware
    if hasattr(app, "middleware") and hasattr(app.middleware, "cls"):
        middleware_info.append({
            "built_middleware_class": str(app.middleware.cls)
        })
    
    # Check if middleware stack was built
    middleware_info.append({
        "middleware_stack_built": hasattr(app, "middleware_stack")
    })
    
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
