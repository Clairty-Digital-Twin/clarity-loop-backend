#!/usr/bin/env python3
"""CLARITY Digital Twin Platform - Application Entry Point.

This file creates the FastAPI application instance using the centralized
factory function from the core container. It is the designated entry point
for running the application locally, ensuring that all middleware and
dependencies are correctly configured, consistent with the Modal deployment.

Usage (from project root):
    uvicorn src.clarity.main:app --reload
"""
from pathlib import Path
import sys

# This is required to ensure that the application can find the 'clarity' module
# when run with `uvicorn src.clarity.main:app` from the project root.
# It adds the project's root directory to the Python path.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from clarity.core.container import create_application

# Create the application instance. This is the single source of truth for the app.
app = create_application()
