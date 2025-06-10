#!/usr/bin/env python3
"""CLARITY Digital Twin Platform - Unified Application Entry Point.

This module serves as the primary entry point for the CLARITY application.
It automatically detects the deployment environment and configures the
appropriate services (AWS, Firebase, or Mock).

## Environment Detection
- Production (AWS): Uses AWS services (Cognito, DynamoDB, S3, SQS/SNS)
- Development: Uses mock services with optional Firebase/AWS
- Testing: Uses mock services exclusively

## Usage
Local development:
    uvicorn src.clarity.main:app --reload

Production:
    gunicorn src.clarity.main:app -c gunicorn.aws.conf.py
"""

from collections.abc import Callable
import os

from fastapi import FastAPI

# Detect deployment environment
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development").lower()
IS_AWS = (
    os.environ.get("AWS_EXECUTION_ENV") is not None
    or os.environ.get("AWS_REGION") is not None
    or os.environ.get("USE_AWS_SERVICES", "false").lower() == "true"
)


def create_app() -> FastAPI:
    """Create FastAPI application with environment-appropriate configuration."""
    if IS_AWS or ENVIRONMENT == "production":
        # Use AWS implementation
        from clarity.main_aws import create_app as create_aws_app
        return create_aws_app()
    # Use original implementation with Firebase/Mock services
    from clarity.core.container import create_application
    return create_application()


# Create the application instance
app = create_app()

# Export for compatibility
get_app = create_app


if __name__ == "__main__":
    import uvicorn

    from clarity.core.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "clarity.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development(),
        log_level=settings.log_level.lower(),
    )
