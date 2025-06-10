"""API v1 Router - Aggregates all v1 endpoints"""

from fastapi import APIRouter

from clarity.api.v1.auth import router as auth_router
from clarity.api.v1.gemini_insights import router as insights_router
from clarity.api.v1.health_data import router as health_data_router
from clarity.api.v1.healthkit_upload import router as healthkit_router
from clarity.api.v1.metrics import router as metrics_router
from clarity.api.v1.pat_analysis import router as pat_router
from clarity.api.v1.websocket.chat_handler import router as websocket_router

# Create the main API router
api_router = APIRouter()

# Include all sub-routers
api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_router.include_router(
    health_data_router, prefix="/health-data", tags=["Health Data"]
)
api_router.include_router(healthkit_router, prefix="/healthkit", tags=["HealthKit"])
api_router.include_router(pat_router, prefix="/pat", tags=["PAT Analysis"])
api_router.include_router(insights_router, prefix="/insights", tags=["AI Insights"])
api_router.include_router(metrics_router, prefix="/metrics", tags=["Metrics"])
api_router.include_router(websocket_router, prefix="/ws", tags=["WebSocket"])
