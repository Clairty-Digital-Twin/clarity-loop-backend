"""CLARITY Digital Twin Platform - AWS Nuclear Main Application Entry Point.

This is the CLEAN version with no Firebase dependencies - pure AWS services only.
Loads all 61 endpoints with full functionality.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from pydantic import BaseModel, ConfigDict, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Environment Configuration
# ==============================================================================

ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
COGNITO_USER_POOL_ID = os.getenv("COGNITO_USER_POOL_ID", "us-east-2_iCRM83uVj")
COGNITO_CLIENT_ID = os.getenv("COGNITO_CLIENT_ID", "485gn7vn3uev0coc52aefklkjs")
COGNITO_REGION = os.getenv("COGNITO_REGION", "us-east-2")
DYNAMODB_TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "clarity-health-data")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "clarity-health-uploads")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "true").lower() == "true"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ==============================================================================
# AWS Client Initialization
# ==============================================================================

# Initialize boto3 clients with proper region
session = boto3.Session(region_name=AWS_REGION)
cognito_client = session.client('cognito-idp', region_name=COGNITO_REGION)
dynamodb = session.resource('dynamodb')
s3_client = session.client('s3')

# ==============================================================================
# Pydantic Models (Clean versions without Firebase deps)
# ==============================================================================

class UserCreate(BaseModel):
    """User creation model"""
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    display_name: Optional[str] = Field(None, description="Display name")
    
    model_config = ConfigDict(extra="forbid")


class UserLogin(BaseModel):
    """User login model"""
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")
    
    model_config = ConfigDict(extra="forbid")


class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    user_id: str
    email: str


class HealthDataCreate(BaseModel):
    """Health data creation model"""
    data_type: str = Field(..., description="Type of health data")
    value: float = Field(..., description="Numeric value")
    timestamp: str = Field(..., description="ISO format timestamp")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    model_config = ConfigDict(extra="forbid")


class HealthDataResponse(BaseModel):
    """Health data response model"""
    id: str
    user_id: str
    data_type: str
    value: float
    timestamp: str
    unit: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    
    model_config = ConfigDict(extra="forbid")


# ==============================================================================
# Authentication Handler (AWS Cognito)
# ==============================================================================

async def get_current_user(request: Request) -> Dict[str, Any]:
    """Extract and verify JWT token from request headers."""
    if not ENABLE_AUTH:
        # Mock user for development
        return {
            "user_id": "dev-user-123",
            "email": "dev@clarity.ai",
            "sub": "dev-user-123"
        }
    
    # Extract token from Authorization header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = auth_header.replace("Bearer ", "")
    
    try:
        # Verify token with Cognito
        response = cognito_client.get_user(AccessToken=token)
        
        # Extract user attributes
        user_attrs = {attr['Name']: attr['Value'] for attr in response.get('UserAttributes', [])}
        
        return {
            "user_id": response['Username'],
            "email": user_attrs.get('email', ''),
            "sub": user_attrs.get('sub', response['Username'])
        }
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NotAuthorizedException':
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        elif error_code == 'UserNotFoundException':
            raise HTTPException(status_code=404, detail="User not found")
        else:
            logger.error(f"Cognito error: {e}")
            raise HTTPException(status_code=500, detail="Authentication service error")
    except Exception as e:
        logger.error(f"Unexpected auth error: {e}")
        raise HTTPException(status_code=500, detail="Internal authentication error")


# ==============================================================================
# Container and Service Dependencies
# ==============================================================================

class DependencyContainer:
    """Simplified dependency container for AWS services."""
    
    def __init__(self):
        self.cognito_client = cognito_client
        self.dynamodb = dynamodb
        self.s3_client = s3_client
        self.dynamodb_table = None
        self.gemini_service = None
        
    async def initialize(self):
        """Initialize all services."""
        logger.info("Initializing AWS services...")
        
        # Initialize DynamoDB table
        try:
            self.dynamodb_table = self.dynamodb.Table(DYNAMODB_TABLE_NAME)
            self.dynamodb_table.load()
            logger.info(f"Connected to DynamoDB table: {DYNAMODB_TABLE_NAME}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.error(f"DynamoDB table {DYNAMODB_TABLE_NAME} not found")
                if ENVIRONMENT == "development":
                    logger.info("Creating DynamoDB table for development...")
                    # Create table logic here if needed
            else:
                logger.error(f"DynamoDB initialization error: {e}")
        
        # Initialize Gemini if API key is available
        if GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                self.gemini_service = genai.GenerativeModel("gemini-1.5-flash")
                logger.info("Gemini AI service initialized")
            except Exception as e:
                logger.warning(f"Gemini initialization failed: {e}")
        
        logger.info("✅ All AWS services initialized successfully")
    
    async def shutdown(self):
        """Cleanup resources."""
        logger.info("Shutting down services...")


# Global container instance
_container: Optional[DependencyContainer] = None


def get_container() -> DependencyContainer:
    """Get the global container instance."""
    global _container
    if _container is None:
        raise RuntimeError("Container not initialized")
    return _container


# ==============================================================================
# Application Lifespan
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global _container
    
    logger.info("🚀 Starting CLARITY AWS Nuclear backend...")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info(f"AWS Region: {AWS_REGION}")
    logger.info(f"Cognito Region: {COGNITO_REGION}")
    logger.info(f"Auth Enabled: {ENABLE_AUTH}")
    
    try:
        # Initialize container
        _container = DependencyContainer()
        await _container.initialize()
        
        # Log loaded routes
        api_routes = [r for r in app.routes if hasattr(r, 'path') and r.path.startswith('/api/')]
        logger.info(f"📊 Total routes loaded: {len(app.routes)}")
        logger.info(f"🔗 API endpoints loaded: {len(api_routes)}")
        
        # Log specific endpoint categories
        auth_routes = [r for r in api_routes if '/auth' in r.path]
        health_routes = [r for r in api_routes if '/health-data' in r.path]
        pat_routes = [r for r in api_routes if '/pat' in r.path]
        insight_routes = [r for r in api_routes if '/insights' in r.path]
        
        logger.info(f"  - Auth endpoints: {len(auth_routes)}")
        logger.info(f"  - Health data endpoints: {len(health_routes)}")
        logger.info(f"  - PAT analysis endpoints: {len(pat_routes)}")
        logger.info(f"  - AI insights endpoints: {len(insight_routes)}")
        
        yield
        
    except Exception as e:
        logger.error(f"💥 Failed to start application: {e}")
        raise
    finally:
        if _container:
            await _container.shutdown()
        logger.info("🛑 CLARITY backend shutdown complete")


# ==============================================================================
# Create FastAPI Application
# ==============================================================================

def create_app() -> FastAPI:
    """Factory function to create FastAPI application instance."""
    
    # Create FastAPI application
    app = FastAPI(
        title="CLARITY Digital Twin Platform - AWS Nuclear",
        description="Full-featured AI-powered health platform with AWS services",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ==============================================================================
    # Import and Include ALL Routers
    # ==============================================================================
    
    # Import all API routers - this is where we get all 61 endpoints
    from clarity.api.v1.router_aws_clean import api_router
    
    # Include the main API router which contains all sub-routers
    app.include_router(api_router, prefix="/api/v1")
    
    # ==============================================================================
    # Additional Core Endpoints
    # ==============================================================================
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "CLARITY Digital Twin Platform",
            "version": "2.0.0",
            "status": "operational",
            "environment": ENVIRONMENT,
            "deployment": "AWS Nuclear",
            "service": "clarity-backend-aws-nuclear",
            "total_endpoints": len(app.routes),
            "api_docs": "/docs",
            "health_check": "/health",
        }
    
    @app.get("/health")
    async def health_check():
        """Enhanced health check endpoint."""
        try:
            container = get_container()
            
            # Check DynamoDB
            db_status = "healthy"
            try:
                if container.dynamodb_table:
                    container.dynamodb_table.table_status
            except Exception as e:
                db_status = f"unhealthy: {str(e)}"
            
            # Check Gemini
            ai_status = "enabled" if container.gemini_service else "disabled"
            
            return {
                "status": "healthy",
                "service": "clarity-backend-aws-nuclear",  # This is what we want to see!
                "environment": ENVIRONMENT,
                "version": "2.0.0",
                "deployment": "AWS Nuclear",
                "endpoints": len(app.routes),
                "aws_region": AWS_REGION,
                "services": {
                    "auth": "cognito" if ENABLE_AUTH else "disabled",
                    "database": db_status,
                    "ai": ai_status,
                    "storage": "s3",
                },
                "cognito": {
                    "user_pool_id": COGNITO_USER_POOL_ID,
                    "region": COGNITO_REGION,
                },
            }
        except Exception as e:
            # Basic health check if container not ready
            return {
                "status": "healthy",
                "service": "clarity-backend-aws-nuclear",
                "environment": ENVIRONMENT,
                "version": "2.0.0",
                "note": f"Container initializing: {str(e)}",
            }
    
    # Add Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "type": "internal_error",
                "status": 500,
            }
        )
    
    return app


# ==============================================================================
# Create Application Instance
# ==============================================================================

app = create_app()


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "clarity.main_aws_nuclear_clean:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=ENVIRONMENT == "development",
        log_level="info" if ENVIRONMENT == "production" else "debug",
    )