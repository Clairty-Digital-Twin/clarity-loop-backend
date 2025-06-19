"""Main API v1 router - AWS Clean version."""

# removed - breaks FastAPI

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends

from clarity.api.v1.auth import router as auth_router
from clarity.api.v1.debug import router as debug_router
from clarity.api.v1.gemini_insights import router as insights_router
from clarity.api.v1.health_data import router as health_data_router
from clarity.api.v1.healthkit_upload import router as healthkit_router
from clarity.api.v1.metrics import router as metrics_router
from clarity.api.v1.pat_analysis import router as pat_router
from clarity.api.v1.test import router as test_router
from clarity.api.v1.websocket.chat_handler import router as websocket_router
from clarity.auth.dependencies import get_current_user

# Configure logging
logger = logging.getLogger(__name__)

# Create main API router
api_router = APIRouter()

# Include all sub-routers
api_router.include_router(
    auth_router,
    prefix="/auth",
    tags=["authentication"],
)

api_router.include_router(
    health_data_router,
    prefix="/health-data",
    tags=["health-data"],
    dependencies=[Depends(get_current_user)],
)

api_router.include_router(
    healthkit_router,
    prefix="/healthkit",
    tags=["healthkit"],
    dependencies=[Depends(get_current_user)],
)

api_router.include_router(
    pat_router,
    prefix="/pat",
    tags=["pat-analysis"],
    dependencies=[Depends(get_current_user)],
)

api_router.include_router(
    insights_router,
    prefix="/insights",
    tags=["ai-insights"],
    dependencies=[Depends(get_current_user)],
)

api_router.include_router(
    metrics_router,
    prefix="/metrics",
    tags=["metrics"],
    dependencies=[Depends(get_current_user)],
)

api_router.include_router(
    websocket_router,
    prefix="/ws",
    tags=["websocket"],
)

# Include debug router in development only
if os.getenv("ENVIRONMENT", "development") == "development":
    api_router.include_router(
        debug_router,
        prefix="/debug",
        tags=["debug"],
    )

api_router.include_router(
    test_router,
    prefix="/test",
    tags=["test"],
)


# Add API info endpoint
@api_router.get("/")
async def api_info() -> dict[str, Any]:
    """Get API information."""
    return {
        "version": "1.0.0",
        "description": "CLARITY Digital Twin Platform API",
        "endpoints": {
            "auth": "/api/v1/auth",
            "health_data": "/api/v1/health-data",
            "healthkit": "/api/v1/healthkit",
            "pat_analysis": "/api/v1/pat",
            "insights": "/api/v1/insights",
            "metrics": "/api/v1/metrics",
            "websocket": "/api/v1/ws",
        },
    }


logger.info("API router configured with %s routes", len(api_router.routes))
