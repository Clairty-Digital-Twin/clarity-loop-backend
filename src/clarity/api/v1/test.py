"""SIMPLE TEST ENDPOINT - NO FANCY IMPORTS."""

# removed - breaks FastAPI

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(tags=["test"])


@router.get("/ping")
async def simple_ping(request: Request) -> dict[str, Any]:
    """Dead simple endpoint to test if anything works."""
    return {
        "message": "PONG! Backend is alive!",
        "path": request.url.path,
        "has_auth_header": "authorization" in request.headers,
        "auth_header_preview": (
            request.headers.get("authorization", "NO AUTH HEADER")[:50]
            if request.headers.get("authorization")
            else "NO AUTH HEADER"
        ),
    }


@router.get("/check-middleware")
async def check_middleware(request: Request) -> dict[str, Any]:
    """Check if middleware sets user."""
    has_user = hasattr(request.state, "user")
    user_info = None

    if has_user and request.state.user:
        user_info = {
            "exists": True,
            "type": str(type(request.state.user)),
            "user_id": getattr(request.state.user, "user_id", "NO USER ID"),
        }

    return {
        "middleware_ran": has_user,
        "user_info": user_info,
        "auth_header": request.headers.get("authorization", "NO AUTH HEADER")[:50],
    }
