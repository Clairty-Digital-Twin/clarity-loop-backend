"""AWS-compatible Clarity backend - CLEAN version with routers only."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging
import os
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from prometheus_client import make_asgi_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE_NAME", "clarity-health-data")
COGNITO_USER_POOL_ID = os.getenv("COGNITO_USER_POOL_ID", "")
COGNITO_CLIENT_ID = os.getenv("COGNITO_CLIENT_ID", "")
COGNITO_REGION = os.getenv("COGNITO_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "clarity-health-uploads")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
API_KEY = os.getenv("CLARITY_API_KEY", "development-key")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "true").lower() == "true"
SKIP_AWS_INIT = os.getenv("SKIP_AWS_INIT", "false").lower() == "true"

# Initialize AWS clients (defer until needed to avoid credential errors)
session = None
dynamodb = None
cognito_client = None
s3_client = None


def init_aws_clients() -> None:
    """Initialize AWS clients when needed."""
    global session, dynamodb, cognito_client, s3_client  # noqa: PLW0603
    if session is None:
        session = boto3.Session(region_name=AWS_REGION)
        dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        cognito_client = session.client("cognito-idp", region_name=COGNITO_REGION)
        s3_client = session.client("s3")


# Initialize Gemini if available
model: Optional[Any] = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)  # type: ignore[attr-defined]
    model = genai.GenerativeModel("gemini-1.5-flash")  # type: ignore[attr-defined]
else:
    logger.warning("GEMINI_API_KEY not set - AI insights will be limited")


# =============================================================================
# Lifespan management
# =============================================================================


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Starting CLARITY Digital Twin backend in %s mode", ENVIRONMENT)
    logger.info("AWS Region: %s", AWS_REGION)
    logger.info("Cognito Region: %s", COGNITO_REGION)
    logger.info("Auth Enabled: %s", ENABLE_AUTH)

    # Initialize DynamoDB table (skip if explicitly disabled or credentials unavailable)
    if SKIP_AWS_INIT:
        logger.info("🔧 AWS initialization skipped via SKIP_AWS_INIT flag")
    else:
        try:
            # Initialize AWS clients first
            init_aws_clients()
            if dynamodb is not None:
                table = dynamodb.Table(DYNAMODB_TABLE)
                table.load()
                logger.info("✅ Connected to DynamoDB table: %s", DYNAMODB_TABLE)
            else:
                logger.warning("DynamoDB client not initialized")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.warning("⚠️  DynamoDB table %s not found", DYNAMODB_TABLE)
            else:
                logger.exception("❌ DynamoDB error")
        except Exception as e:
            # Handle credentials errors and other AWS connectivity issues
            if "NoCredentialsError" in str(e) or "Unable to locate credentials" in str(e):
                logger.warning("🔧 Development mode: AWS credentials not available - running in local mode")
            else:
                logger.warning("⚠️  AWS connection issue (continuing in local mode): %s", str(e))

    yield

    logger.info("Shutting down CLARITY backend")


# =============================================================================
# Create FastAPI app
# =============================================================================


app = FastAPI(
    title="CLARITY Digital Twin Platform",
    description="Production AWS-native health data platform with comprehensive API endpoints",
    version="0.2.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Include ALL API routers
# =============================================================================

# Import the CLEAN AWS router - no duplicates
from clarity.api.v1.router import api_router as v1_router  # noqa: E402

# Include ONLY the clean router - professional single source of truth
app.include_router(v1_router, prefix="/api/v1", tags=["API v1"])

logger.info("✅ Included CLEAN router - professional endpoint structure")

# =============================================================================
# App Factory for Testing
# =============================================================================


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app (used by tests)."""
    return app


def get_app() -> FastAPI:
    """Get the FastAPI app instance (used by tests)."""
    return app


# =============================================================================
# Core Endpoints
# =============================================================================


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint."""
    return {
        "name": "CLARITY Digital Twin Platform",
        "version": "0.2.0",
        "status": "operational",
        "service": "clarity-backend-aws-full",
        "environment": ENVIRONMENT,
        "deployment": "AWS Production",
        "total_endpoints": len(app.routes),
        "api_docs": "/docs",
    }


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "clarity-backend-aws-full",
        "environment": ENVIRONMENT,
        "version": "0.2.0",
        "features": {
            "cognito_auth": ENABLE_AUTH,
            "api_key_auth": bool(API_KEY),
            "dynamodb": bool(DYNAMODB_TABLE),
            "gemini_insights": bool(model),
        },
    }


# Add Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
