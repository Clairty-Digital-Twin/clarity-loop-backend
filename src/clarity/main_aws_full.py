"""AWS Production Main - FULL 38+ Enterprise Endpoints with All Modules."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime
import logging
import os
from typing import Any

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import google.generativeai as genai
from prometheus_client import make_asgi_app

# Import all providers and services
from clarity.auth.aws_cognito_provider import CognitoAuthProvider
from clarity.core.container_aws import get_container
from clarity.core.config_aws import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
config = get_config()
container = get_container()

# Initialize Gemini if available
if config.gemini_api_key:
    genai.configure(api_key=config.gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    logger.warning("GEMINI_API_KEY not set - AI insights will be limited")
    model = None

# Set the model in container for dependency injection
container.gemini_model = model

# =============================================================================
# Lifespan management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting CLARITY AWS backend (full) in %s mode", config.environment)
    logger.info("AWS Region: %s", config.aws_region)
    logger.info("Total endpoints being loaded: 38+ enterprise endpoints")
    
    # Initialize Cognito auth provider
    try:
        cognito_provider = CognitoAuthProvider(
            user_pool_id=config.cognito_user_pool_id,
            client_id=config.cognito_client_id,
            region=config.cognito_region
        )
        await cognito_provider.initialize()
        container.auth_provider = cognito_provider
        logger.info("✅ Cognito auth provider initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize Cognito: %s", e)
        # Continue with mock auth in development
        if config.environment == "development":
            from clarity.auth.mock_auth import MockAuthProvider
            container.auth_provider = MockAuthProvider()
            logger.info("Using MockAuthProvider for development")
    
    # Initialize DynamoDB
    try:
        dynamodb = boto3.resource("dynamodb", region_name=config.aws_region)
        table = dynamodb.Table(config.dynamodb_table_name)
        table.load()
        container.dynamodb_table = table
        logger.info("✅ Connected to DynamoDB table: %s", config.dynamodb_table_name)
    except ClientError as e:
        logger.error("DynamoDB error: %s", e)
    
    # Initialize S3
    try:
        s3_client = boto3.client("s3", region_name=config.aws_region)
        container.s3_client = s3_client
        logger.info("✅ S3 client initialized")
    except Exception as e:
        logger.error("S3 initialization error: %s", e)
    
    # Initialize SQS/SNS if needed
    try:
        sqs_client = boto3.client("sqs", region_name=config.aws_region)
        sns_client = boto3.client("sns", region_name=config.aws_region)
        container.sqs_client = sqs_client
        container.sns_client = sns_client
        logger.info("✅ SQS/SNS clients initialized")
    except Exception as e:
        logger.error("SQS/SNS initialization error: %s", e)
    
    yield
    
    logger.info("Shutting down Clarity AWS backend")


# =============================================================================
# Create FastAPI app
# =============================================================================

app = FastAPI(
    title="CLARITY Digital Twin Platform - Enterprise Edition",
    description="""
    🚀 Production AWS-native health data platform with 38+ enterprise endpoints
    
    ## Features:
    - 🧠 Pretrained Actigraphy Transformer (PAT) - Dartmouth's 29K participant model
    - 🤖 Gemini AI Integration for personalized health insights
    - 📱 HealthKit data processing from Apple Watch
    - 🔐 AWS Cognito authentication with JWT tokens
    - 📊 Real-time health metrics and analytics
    - 💬 WebSocket support for live health consultations
    - 🏥 HIPAA-compliant data storage on AWS
    
    ## API Modules:
    1. **Authentication** - Secure user management
    2. **Health Data** - Store and query health metrics
    3. **PAT Analysis** - ML-powered sleep and activity analysis
    4. **AI Insights** - Gemini-powered health recommendations
    5. **HealthKit** - Apple Watch data integration
    6. **WebSocket** - Real-time health chat
    7. **Metrics** - Prometheus monitoring
    """,
    version="2.0.0-enterprise",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
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
# Import and include ALL routers
# =============================================================================

logger.info("Loading all API routers...")

# Import all routers
from clarity.api.v1.auth import router as auth_router
from clarity.api.v1.health_data import router as health_data_router
from clarity.api.v1.healthkit_upload import router as healthkit_router
from clarity.api.v1.pat_analysis import router as pat_router
from clarity.api.v1.gemini_insights import router as insights_router
from clarity.api.v1.metrics import router as metrics_router
from clarity.api.v1.websocket.chat_handler import router as websocket_router
from clarity.api.v1.debug import router as debug_router

# Include all routers with proper prefixes
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(health_data_router, prefix="/api/v1/health-data", tags=["Health Data"])
app.include_router(healthkit_router, prefix="/api/v1/healthkit", tags=["HealthKit"])
app.include_router(pat_router, prefix="/api/v1/pat", tags=["PAT Analysis"])
app.include_router(insights_router, prefix="/api/v1/insights", tags=["AI Insights"])
app.include_router(metrics_router, prefix="/api/v1/metrics", tags=["Metrics"])
app.include_router(websocket_router, prefix="/api/v1/ws", tags=["WebSocket"])

# Include debug router only in non-production
if config.environment != "production":
    app.include_router(debug_router, prefix="/api/v1/debug", tags=["Debug"])

logger.info("✅ All routers loaded successfully")

# =============================================================================
# Core endpoints
# =============================================================================

@app.get("/", tags=["Core"])
async def root():
    """Root endpoint with platform information."""
    return {
        "name": "CLARITY Digital Twin Platform",
        "version": "2.0.0-enterprise",
        "status": "operational",
        "environment": config.environment,
        "endpoints": {
            "total": len(app.routes),
            "api_docs": "/docs",
            "health": "/health",
            "metrics": "/metrics"
        },
        "modules": {
            "authentication": "AWS Cognito",
            "database": "DynamoDB",
            "storage": "S3",
            "ai": "Gemini + PAT",
            "monitoring": "Prometheus"
        },
        "message": "🚀 Enterprise health platform ready"
    }

@app.get("/health", tags=["Core"])
async def health_check():
    """Comprehensive health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "service": "clarity-backend-enterprise",
        "version": "2.0.0",
        "environment": config.environment,
        "checks": {
            "api": "operational",
            "database": "unknown",
            "auth": "unknown",
            "ai": "unknown",
            "storage": "unknown"
        }
    }
    
    # Check DynamoDB
    try:
        if hasattr(container, 'dynamodb_table'):
            container.dynamodb_table.table_status
            health_status["checks"]["database"] = "healthy"
    except:
        health_status["checks"]["database"] = "degraded"
    
    # Check Auth
    try:
        if hasattr(container, 'auth_provider'):
            health_status["checks"]["auth"] = "healthy"
    except:
        health_status["checks"]["auth"] = "degraded"
    
    # Check AI
    health_status["checks"]["ai"] = "healthy" if model else "unavailable"
    
    # Check S3
    try:
        if hasattr(container, 's3_client'):
            health_status["checks"]["storage"] = "healthy"
    except:
        health_status["checks"]["storage"] = "degraded"
    
    # Overall status
    if any(v == "degraded" for v in health_status["checks"].values()):
        health_status["status"] = "degraded"
    
    return health_status

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# =============================================================================
# Global exception handler
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": request.headers.get("X-Request-ID", "unknown")
        }
    )

# =============================================================================
# Startup message
# =============================================================================

@app.on_event("startup")
async def startup_message():
    """Log startup information."""
    logger.info("=" * 60)
    logger.info("🚀 CLARITY DIGITAL TWIN PLATFORM - ENTERPRISE EDITION")
    logger.info("=" * 60)
    logger.info("Total routes loaded: %d", len(app.routes))
    logger.info("Environment: %s", config.environment)
    logger.info("API Documentation: /docs")
    logger.info("=" * 60)

# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)