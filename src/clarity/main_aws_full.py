"""
AWS-compatible Clarity backend - Full version with Cognito
Incrementally adding features back
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, Depends, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import boto3
from botocore.exceptions import ClientError
import google.generativeai as genai

from clarity.auth.aws_cognito_provider import get_cognito_provider
from clarity.models.user import User

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE", "clarity-health-data")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
API_KEY = os.getenv("CLARITY_API_KEY", "development-key")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DYNAMODB_ENDPOINT = os.getenv("DYNAMODB_ENDPOINT", None)

# Cognito configuration
COGNITO_USER_POOL_ID = os.getenv("COGNITO_USER_POOL_ID", "")
COGNITO_CLIENT_ID = os.getenv("COGNITO_CLIENT_ID", "")

# Initialize AWS clients
if DYNAMODB_ENDPOINT:
    dynamodb = boto3.resource(
        'dynamodb', 
        region_name=AWS_REGION,
        endpoint_url=DYNAMODB_ENDPOINT
    )
else:
    dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)

# Initialize Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    logger.warning("GEMINI_API_KEY not set - insights will be unavailable")
    model = None

# Security
security = HTTPBearer(auto_error=False)


# Pydantic models
class HealthDataInput(BaseModel):
    """Input model for health data"""
    data_type: str = Field(..., description="Type of health data")
    value: float = Field(..., description="Numeric value")
    timestamp: str = Field(..., description="ISO format timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class AuthRequest(BaseModel):
    """Authentication request"""
    email: str = Field(..., description="User email")
    password: str = Field(..., description="User password")


class AuthResponse(BaseModel):
    """Authentication response"""
    success: bool
    user_id: Optional[str] = None
    email: Optional[str] = None
    tokens: Optional[Dict[str, str]] = None
    error: Optional[str] = None


class InsightRequest(BaseModel):
    """Request model for generating insights"""
    query: str = Field(..., description="Health-related question")
    include_recent_data: bool = Field(default=True)


# Authentication dependencies
async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """Get current user from JWT token or API key"""
    
    # Check API key first (backwards compatibility)
    if x_api_key and x_api_key == API_KEY:
        return {"user_id": "api-key-user", "auth_type": "api_key"}
    
    # Check Cognito JWT token
    if credentials and COGNITO_USER_POOL_ID:
        try:
            cognito = get_cognito_provider()
            claims = await cognito.verify_token(credentials.credentials)
            if claims:
                return {
                    "user_id": claims.get("sub"),
                    "email": claims.get("email"),
                    "auth_type": "cognito"
                }
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
    
    raise HTTPException(status_code=401, detail="Invalid authentication")


# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info(f"Starting Clarity AWS backend (full) in {ENVIRONMENT} mode")
    
    # Initialize Cognito if configured
    if COGNITO_USER_POOL_ID:
        try:
            cognito = get_cognito_provider()
            await cognito.initialize()
            logger.info("Cognito authentication enabled")
        except Exception as e:
            logger.error(f"Failed to initialize Cognito: {e}")
            if ENVIRONMENT == "production":
                raise
    
    # Initialize DynamoDB table
    try:
        table = dynamodb.Table(DYNAMODB_TABLE)
        table.load()
        logger.info(f"Connected to DynamoDB table: {DYNAMODB_TABLE}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            logger.warning(f"DynamoDB table {DYNAMODB_TABLE} not found")
            if ENVIRONMENT == "development":
                # Create table in development
                try:
                    table = dynamodb.create_table(
                        TableName=DYNAMODB_TABLE,
                        KeySchema=[
                            {'AttributeName': 'user_id', 'KeyType': 'HASH'},
                            {'AttributeName': 'data_id', 'KeyType': 'RANGE'}
                        ],
                        AttributeDefinitions=[
                            {'AttributeName': 'user_id', 'AttributeType': 'S'},
                            {'AttributeName': 'data_id', 'AttributeType': 'S'}
                        ],
                        BillingMode='PAY_PER_REQUEST'
                    )
                    table.wait_until_exists()
                    logger.info("Created DynamoDB table")
                except Exception as create_error:
                    logger.error(f"Failed to create table: {create_error}")
        else:
            logger.error(f"DynamoDB error: {e}")
    
    yield
    
    logger.info("Shutting down Clarity AWS backend")


# Create FastAPI app
app = FastAPI(
    title="Clarity Health Backend (AWS Full)",
    description="AWS-native health data backend with Cognito authentication",
    version="0.2.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint (no auth)
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "service": "clarity-backend-aws-full",
        "environment": ENVIRONMENT,
        "version": "0.2.0",
        "features": {
            "cognito_auth": bool(COGNITO_USER_POOL_ID),
            "api_key_auth": True,
            "dynamodb": True,
            "gemini_insights": bool(GEMINI_API_KEY)
        }
    }


# Authentication endpoints
@app.post("/api/v1/auth/signup", response_model=AuthResponse)
async def signup(auth_data: AuthRequest):
    """Create new user account"""
    if not COGNITO_USER_POOL_ID:
        return AuthResponse(
            success=False,
            error="Authentication service not configured"
        )
    
    try:
        cognito = get_cognito_provider()
        user = await cognito.create_user(
            email=auth_data.email,
            password=auth_data.password
        )
        
        if user:
            # Auto-login after signup
            tokens = await cognito.authenticate(
                email=auth_data.email,
                password=auth_data.password
            )
            
            return AuthResponse(
                success=True,
                user_id=user.uid,
                email=user.email,
                tokens=tokens
            )
        
        return AuthResponse(success=False, error="Failed to create user")
        
    except Exception as e:
        logger.error(f"Signup failed: {e}")
        return AuthResponse(success=False, error=str(e))


@app.post("/api/v1/auth/login", response_model=AuthResponse)
async def login(auth_data: AuthRequest):
    """Authenticate user and get tokens"""
    if not COGNITO_USER_POOL_ID:
        return AuthResponse(
            success=False,
            error="Authentication service not configured"
        )
    
    try:
        cognito = get_cognito_provider()
        tokens = await cognito.authenticate(
            email=auth_data.email,
            password=auth_data.password
        )
        
        if tokens:
            # Get user details from token
            claims = await cognito.verify_token(tokens['id_token'])
            
            return AuthResponse(
                success=True,
                user_id=claims.get('sub'),
                email=claims.get('email'),
                tokens=tokens
            )
        
        return AuthResponse(success=False, error="Invalid credentials")
        
    except Exception as e:
        logger.error(f"Login failed: {e}")
        return AuthResponse(success=False, error=str(e))


# Store health data endpoint
@app.post("/api/v1/health-data")
async def store_health_data(
    data: HealthDataInput,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Store health data in DynamoDB"""
    import uuid
    from datetime import datetime
    
    try:
        # Use authenticated user ID
        user_id = current_user["user_id"]
        
        # Generate unique data ID
        data_id = f"{data.data_type}_{uuid.uuid4().hex[:8]}_{datetime.utcnow().timestamp()}"
        
        # Prepare item for DynamoDB
        item = {
            'user_id': user_id,
            'data_id': data_id,
            'data_type': data.data_type,
            'value': str(data.value),
            'timestamp': data.timestamp,
            'created_at': datetime.utcnow().isoformat(),
            'metadata': data.metadata or {},
            'auth_type': current_user.get('auth_type', 'unknown')
        }
        
        # Store in DynamoDB
        table = dynamodb.Table(DYNAMODB_TABLE)
        table.put_item(Item=item)
        
        logger.info(f"Stored health data: {data_id} for user {user_id}")
        
        return {
            "success": True,
            "message": "Health data stored successfully",
            "data_id": data_id,
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Failed to store health data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get user's health data
@app.get("/api/v1/health-data")
async def get_user_health_data(
    limit: int = 50,
    data_type: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Retrieve user's health data"""
    try:
        user_id = current_user["user_id"]
        table = dynamodb.Table(DYNAMODB_TABLE)
        
        # Query user's data
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key('user_id').eq(user_id),
            Limit=limit,
            ScanIndexForward=False  # Most recent first
        )
        
        items = response.get('Items', [])
        
        # Filter by data type if specified
        if data_type:
            items = [item for item in items if item.get('data_type') == data_type]
        
        return {
            "success": True,
            "user_id": user_id,
            "data": items,
            "count": len(items)
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Generate insights endpoint
@app.post("/api/v1/insights")
async def generate_insights(
    request: InsightRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate health insights using Gemini API"""
    
    if not model:
        return {
            "success": False,
            "error": "Gemini API not configured"
        }
    
    try:
        user_id = current_user["user_id"]
        context = f"User asks: {request.query}"
        
        # Fetch recent health data for context
        if request.include_recent_data:
            try:
                table = dynamodb.Table(DYNAMODB_TABLE)
                response = table.query(
                    KeyConditionExpression=boto3.dynamodb.conditions.Key('user_id').eq(user_id),
                    Limit=10,
                    ScanIndexForward=False
                )
                
                if response.get('Items'):
                    recent_data = []
                    for item in response['Items']:
                        recent_data.append(
                            f"{item['data_type']}: {item['value']} at {item['timestamp']}"
                        )
                    context += f"\n\nRecent health data:\n" + "\n".join(recent_data)
                    
            except Exception as e:
                logger.warning(f"Could not fetch recent data: {e}")
        
        # Generate insight
        prompt = f"""You are a helpful health assistant. Based on the following context, 
        provide a brief, actionable health insight. Be encouraging and focus on positive 
        actions the user can take.
        
        {context}
        
        Provide a concise response (2-3 sentences max)."""
        
        response = model.generate_content(prompt)
        
        return {
            "success": True,
            "insight": response.text,
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Failed to generate insight: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# User profile endpoint
@app.get("/api/v1/user/profile")
async def get_user_profile(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get user profile information"""
    if current_user.get("auth_type") == "api_key":
        return {
            "user_id": current_user["user_id"],
            "auth_type": "api_key",
            "message": "API key authentication - no profile available"
        }
    
    try:
        cognito = get_cognito_provider()
        user = await cognito.get_user(current_user["user_id"])
        
        if user:
            return {
                "user_id": user.uid,
                "email": user.email,
                "display_name": user.display_name,
                "created_at": user.created_at,
                "metadata": user.metadata
            }
        
        raise HTTPException(status_code=404, detail="User not found")
        
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)