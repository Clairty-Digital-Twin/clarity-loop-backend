"""CLARITY Digital Twin Platform - API v1 Package.

Version 1 of the CLARITY platform API endpoints.
This module contains all the route definitions for the first version of the API.

Routes are now managed through router.py for production deployment.
Legacy router imports have been removed to avoid conflicts.

Available routers:
- auth: Authentication and user management endpoints
- health_data: Health data upload and management endpoints
- pat_analysis: PAT (Pretrained Actigraphy Transformer) analysis endpoints
- gemini_insights: Gemini AI health insights generation endpoints
- websocket: Real-time WebSocket communication
- metrics: Health metrics calculation and aggregation
- test: Basic connectivity and health tests
- debug: Development and debugging endpoints (dev only)
"""

# removed - breaks FastAPI

__version__ = "1.0.0"

# Clean API - routers are imported directly where needed
# This avoids circular imports and conflicting legacy router definitions

__all__ = [
    "__version__",
]
