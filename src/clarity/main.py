#!/usr/bin/env python3
"""CLARITY Digital Twin Platform - Application Entry Point.

This file creates the FastAPI application instance using the centralized
factory function from the core container. It is the designated entry point
for running the application locally, ensuring that all middleware and
dependencies are correctly configured, consistent with the Modal deployment.

Usage (from project root):
    uvicorn src.clarity.main:app --reload
"""
import os

from clarity.core.container import create_application

# Check if we should use AWS implementation for testing
USE_AWS_IMPL = os.environ.get("USE_AWS_IMPL", "").lower() == "true"

if USE_AWS_IMPL:
    # Import AWS implementation for testing
    from clarity.main_aws import create_app
else:
    # Use original implementation with compatibility wrapper
    def create_app():
        """Compatibility wrapper for tests expecting create_app function."""
        return create_application()


# Create the application instance. This is the single source of truth for the app.
app = create_application() if not USE_AWS_IMPL else create_app()

# Compatibility alias for tests expecting get_app
get_app = create_app
