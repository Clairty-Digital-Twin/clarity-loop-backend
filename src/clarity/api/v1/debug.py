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
