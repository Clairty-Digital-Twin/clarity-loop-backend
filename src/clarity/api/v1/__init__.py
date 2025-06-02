"""CLARITY Digital Twin Platform - API v1 Package.

Version 1 of the CLARITY platform API endpoints.
This module contains all the route definitions for the first version of the API.

Routes:
- health_data: Health data upload and management endpoints
- pat_analysis: PAT (Pretrained Actigraphy Transformer) analysis endpoints
"""

from fastapi import APIRouter

from clarity.api.v1.health_data import router as health_data_router
from clarity.api.v1.pat_analysis import router as pat_analysis_router

# Create the main v1 router and include all sub-routers
router = APIRouter(prefix="/api/v1", tags=["v1"])

# Include health data routes
router.include_router(health_data_router)

# Include PAT analysis routes
router.include_router(pat_analysis_router)

__all__ = [
    "router",
]
