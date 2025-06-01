"""CLARITY Digital Twin Platform - API Package

This module contains the FastAPI application and all API route definitions
for the psychiatry and mental health digital twin platform.

The API is organized into versioned modules:
- v1/: Version 1 API endpoints
"""

__version__ = "0.1.0"
__author__ = "CLARITY Digital Twin Platform"

# Export main API components for easy importing
from .v1 import router as v1_router

__all__ = [
    "v1_router",
]
