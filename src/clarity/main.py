"""AWS-compatible Clarity backend - CLEAN version with routers only."""

# removed – breaks FastAPI

from contextlib import asynccontextmanager
import logging
import os
import sys
from typing import TYPE_CHECKING, Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from prometheus_client import make_asgi_app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# Configure logging (will be enhanced by observability setup)
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
    global session, dynamodb, cognito_client, s3_client  # noqa: PLW0603 - Singleton pattern for AWS service clients
    if session is None:
        session = boto3.Session(region_name=AWS_REGION)
        dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        cognito_client = session.client("cognito-idp", region_name=COGNITO_REGION)
        s3_client = session.client("s3")


# Initialize Gemini if available
model: Any | None = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)  # type: ignore[attr-defined]
    model = genai.GenerativeModel("gemini-1.5-flash")  # type: ignore[attr-defined]
else:
    logger.warning("GEMINI_API_KEY not set - AI insights will be limited")


# =============================================================================
# Lifespan management
# =============================================================================


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: RUF029
    """Application lifespan manager."""
    logger.info("Starting CLARITY Digital Twin backend in %s mode", ENVIRONMENT)
    logger.info("AWS Region: %s", AWS_REGION)
    logger.info("Cognito Region: %s", COGNITO_REGION)
    logger.info("Auth Enabled: %s", ENABLE_AUTH)
    
    # Setup comprehensive observability stack
    from clarity.observability import setup_observability
    from clarity.observability.logging import setup_structured_logging
    from clarity.observability.metrics import get_metrics
    from clarity.observability.alerting import get_alert_manager, SlackNotificationChannel
    
    # Setup structured logging
    setup_structured_logging(
        level=os.getenv("LOG_LEVEL", "INFO"),
        json_logs=ENVIRONMENT == "production",
        enable_rich=ENVIRONMENT == "development",
    )
    logger.info("✅ Structured logging configured")
    
    # Setup OpenTelemetry instrumentation
    setup_observability(
        service_name="clarity-backend",
        service_version="0.2.0",
        environment=ENVIRONMENT,
        jaeger_endpoint=os.getenv("JAEGER_ENDPOINT"),
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        enable_console_export=ENVIRONMENT == "development",
    )
    logger.info("✅ OpenTelemetry instrumentation configured")
    
    # Initialize metrics and set app info
    metrics = get_metrics()
    metrics.set_app_info(
        version="0.2.0",
        environment=ENVIRONMENT,
        aws_region=AWS_REGION,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    logger.info("✅ Metrics collection initialized")
    
    # Setup alerting channels
    alert_manager = get_alert_manager()
    
    # Add Slack notification if webhook is configured
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    if slack_webhook:
        slack_channel = SlackNotificationChannel(
            name="slack_alerts",
            webhook_url=slack_webhook,
            channel=os.getenv("SLACK_CHANNEL", "#alerts")
        )
        alert_manager.add_notification_channel(slack_channel)
        logger.info("✅ Slack alerts configured")
    
    logger.info("🔍 Clarity Observability Stack fully initialized")

    # Initialize lockout service with Redis URL from environment
    from clarity.auth.lockout_service import get_lockout_service
    lockout_service = get_lockout_service()
    logger.info("✅ Account lockout service initialized")

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
        except NoCredentialsError:
            logger.warning(
                "🔧 Development mode: AWS credentials not available - running in local mode"
            )
        except BotoCoreError as e:
            # Handle other AWS connectivity issues
            if "Unable to locate credentials" in str(e):
                logger.warning(
                    "🔧 Development mode: AWS credentials not available - running in local mode"
                )
            else:
                logger.warning(
                    "⚠️  AWS connection issue (continuing in local mode): %s", str(e)
                )

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
    openapi_tags=[
        {
            "name": "authentication",
            "description": "User authentication and authorization endpoints"
        },
        {
            "name": "health-data",
            "description": "Health data management and retrieval"
        },
        {
            "name": "healthkit",
            "description": "Apple HealthKit data integration"
        },
        {
            "name": "pat-analysis",
            "description": "Physical Activity Test (PAT) analysis endpoints"
        },
        {
            "name": "ai-insights",
            "description": "AI-powered health insights generation"
        },
        {
            "name": "metrics",
            "description": "Health metrics and statistics"
        },
        {
            "name": "websocket",
            "description": "WebSocket real-time communication"
        },
        {
            "name": "debug",
            "description": "Debug endpoints (development only)"
        },
        {
            "name": "test",
            "description": "Test endpoints for API validation"
        }
    ],
    servers=[
        {
            "url": "http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com",
            "description": "Production server (AWS ALB)"
        },
        {
            "url": "http://localhost:8000",
            "description": "Local development server"
        }
    ],
    contact={
        "name": "CLARITY Support",
        "email": "support@clarity.novamindnyc.com",
        "url": "https://clarity.novamindnyc.com"
    },
    license_info={
        "name": "Proprietary",
        "url": "https://clarity.novamindnyc.com/license"
    }
)

# Add CORS middleware with SECURE configuration - NO WILDCARDS
from clarity.core.config import get_settings

settings = get_settings()

app.add_middleware(
    CORSMiddleware,
            allow_origins=settings.get_cors_origins,                     # ✅ EXPLICIT ORIGINS ONLY
    allow_credentials=settings.cors_allow_credentials,            # ✅ SAFE WITH EXPLICIT ORIGINS  
    allow_methods=settings.cors_allowed_methods,                  # ✅ SPECIFIC METHODS ONLY
    allow_headers=settings.cors_allowed_headers,                  # ✅ SPECIFIC HEADERS ONLY
    max_age=settings.cors_max_age,                               # ✅ CACHE PREFLIGHT REQUESTS
)

logger.info("🔒 CORS Security: Hardened configuration applied - NO wildcards allowed")

# Add request size limiter middleware - PREVENT DoS ATTACKS
from clarity.middleware.request_size_limiter import RequestSizeLimiterMiddleware

app.add_middleware(
    RequestSizeLimiterMiddleware,
    max_request_size=settings.max_request_size,          # ✅ 10MB general limit
    max_json_size=settings.max_json_size,                # ✅ 5MB JSON limit  
    max_upload_size=settings.max_upload_size,            # ✅ 50MB upload limit
    max_form_size=settings.max_form_size,                # ✅ 1MB form limit
)

logger.info("🔒 Request Size Limiter: DoS protection active - payload limits enforced")

# Add timeout middleware to prevent hangs from incomplete uploads
from starlette.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Configure based on environment in production
)

# Note: Timeout middleware should be added at uvicorn level for better control
# uvicorn src.clarity.main:app --timeout-keep-alive 15 --timeout-graceful-shutdown 5

# Add security headers middleware
from clarity.middleware.security_headers import SecurityHeadersMiddleware

app.add_middleware(
    SecurityHeadersMiddleware,
    enable_hsts=True,  # Enforce HTTPS with HSTS
    enable_csp=True,   # Enable Content Security Policy
    cache_control="no-store, private",  # Prevent caching of sensitive data
)
logger.info("✅ Added security headers middleware")

# Add authentication middleware
from clarity.middleware.auth_middleware import CognitoAuthMiddleware

app.add_middleware(CognitoAuthMiddleware)
logger.info("✅ Added authentication middleware")

# Add rate limiting middleware
from clarity.middleware.rate_limiting import setup_rate_limiting

# Get Redis URL from environment for distributed rate limiting
redis_url = os.getenv("REDIS_URL")
limiter = setup_rate_limiting(app, redis_url=redis_url)
logger.info("✅ Added rate limiting middleware")

# Add observability middleware
from clarity.observability.middleware import ObservabilityMiddleware, SystemMetricsMiddleware
from clarity.observability.correlation import CorrelationMiddleware

app.add_middleware(ObservabilityMiddleware)
app.add_middleware(SystemMetricsMiddleware, update_interval=30)
app.add_middleware(CorrelationMiddleware)
logger.info("✅ Added comprehensive observability middleware")

# Add request logging middleware in development
if ENVIRONMENT == "development":
    from clarity.middleware.request_logger import RequestLoggingMiddleware

    app.add_middleware(RequestLoggingMiddleware)
    logger.info("✅ Added request logging middleware for development")

# =============================================================================
# Include ALL API routers
# =============================================================================

# Import the CLEAN AWS router - no duplicates
from clarity.api.v1.router import api_router as v1_router  # noqa: E402
from clarity.core.openapi import custom_openapi  # noqa: E402

# Include ONLY the clean router - professional single source of truth
app.include_router(v1_router, prefix="/api/v1")

# Set custom OpenAPI schema
app.openapi = lambda: custom_openapi(app)  # type: ignore[method-assign]

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
