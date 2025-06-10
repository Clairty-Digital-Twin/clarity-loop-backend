"""
AWS-compatible Clarity backend - simplified version
No Firebase, no Google Cloud dependencies
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import boto3
from botocore.exceptions import ClientError
import google.generativeai as genai

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

# Initialize AWS clients
if DYNAMODB_ENDPOINT:
    # Local DynamoDB for development
    dynamodb = boto3.resource(
        'dynamodb', 
        region_name=AWS_REGION,
        endpoint_url=DYNAMODB_ENDPOINT
    )
else:
    # Real AWS DynamoDB
    dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)

# Initialize Gemini (direct API, not Vertex AI)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    logger.warning("GEMINI_API_KEY not set - insights will be unavailable")
    model = None


# Pydantic models
class HealthDataInput(BaseModel):
    """Input model for health data"""
    user_id: str = Field(..., description="User identifier")
    data_type: str = Field(..., description="Type of health data (e.g., 'heart_rate', 'steps')")
    value: float = Field(..., description="Numeric value")
    timestamp: str = Field(..., description="ISO format timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class HealthDataResponse(BaseModel):
    """Response model for health data operations"""
    success: bool
    message: str
    data_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class InsightRequest(BaseModel):
    """Request model for generating insights"""
    user_id: str = Field(..., description="User identifier")
    query: str = Field(..., description="Health-related question or prompt")
    include_recent_data: bool = Field(default=True, description="Include recent health data in context")


class InsightResponse(BaseModel):
    """Response model for insights"""
    success: bool
    insight: Optional[str] = None
    error: Optional[str] = None


# Simple API key authentication
async def verify_api_key(x_api_key: str = Header(...)) -> str:
    """Simple API key verification"""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info(f"Starting Clarity AWS backend in {ENVIRONMENT} mode")
    
    # Initialize DynamoDB table if needed
    try:
        table = dynamodb.Table(DYNAMODB_TABLE)
        table.load()
        logger.info(f"Connected to DynamoDB table: {DYNAMODB_TABLE}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            logger.warning(f"DynamoDB table {DYNAMODB_TABLE} not found - creating...")
            # In production, table should be created via CloudFormation/Terraform
            if ENVIRONMENT == "development":
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
    title="Clarity Health Backend (AWS)",
    description="Simplified AWS-compatible health data backend",
    version="0.1.0",
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


# Health check endpoint (no auth required)
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "service": "clarity-backend-aws",
        "environment": ENVIRONMENT,
        "version": "0.1.0"
    }


# Store health data endpoint
@app.post("/api/v1/data", response_model=HealthDataResponse)
async def store_health_data(
    data: HealthDataInput,
    api_key: str = Depends(verify_api_key)
):
    """Store health data in DynamoDB"""
    import uuid
    from datetime import datetime
    
    try:
        # Generate unique data ID
        data_id = f"{data.data_type}_{uuid.uuid4().hex[:8]}_{datetime.utcnow().timestamp()}"
        
        # Prepare item for DynamoDB
        item = {
            'user_id': data.user_id,
            'data_id': data_id,
            'data_type': data.data_type,
            'value': str(data.value),  # DynamoDB stores numbers as strings
            'timestamp': data.timestamp,
            'created_at': datetime.utcnow().isoformat(),
            'metadata': data.metadata or {}
        }
        
        # Store in DynamoDB
        table = dynamodb.Table(DYNAMODB_TABLE)
        table.put_item(Item=item)
        
        logger.info(f"Stored health data: {data_id} for user {data.user_id}")
        
        return HealthDataResponse(
            success=True,
            message="Health data stored successfully",
            data_id=data_id,
            data=item
        )
        
    except Exception as e:
        logger.error(f"Failed to store health data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Generate insights endpoint
@app.post("/api/v1/insights", response_model=InsightResponse)
async def generate_insights(
    request: InsightRequest,
    api_key: str = Depends(verify_api_key)
):
    """Generate health insights using Gemini API"""
    
    if not model:
        return InsightResponse(
            success=False,
            error="Gemini API not configured - please set GEMINI_API_KEY"
        )
    
    try:
        context = f"User {request.user_id} asks: {request.query}"
        
        # Optionally fetch recent health data for context
        if request.include_recent_data:
            try:
                table = dynamodb.Table(DYNAMODB_TABLE)
                response = table.query(
                    KeyConditionExpression=boto3.dynamodb.conditions.Key('user_id').eq(request.user_id),
                    Limit=10,
                    ScanIndexForward=False  # Most recent first
                )
                
                if response.get('Items'):
                    recent_data = []
                    for item in response['Items']:
                        recent_data.append(f"{item['data_type']}: {item['value']} at {item['timestamp']}")
                    context += f"\n\nRecent health data:\n" + "\n".join(recent_data)
                    
            except Exception as e:
                logger.warning(f"Could not fetch recent data: {e}")
        
        # Generate insight with Gemini
        prompt = f"""You are a helpful health assistant. Based on the following context, provide a brief, 
        actionable health insight. Be encouraging and focus on positive actions the user can take.
        
        {context}
        
        Provide a concise response (2-3 sentences max)."""
        
        response = model.generate_content(prompt)
        
        return InsightResponse(
            success=True,
            insight=response.text
        )
        
    except Exception as e:
        logger.error(f"Failed to generate insight: {e}")
        return InsightResponse(
            success=False,
            error=f"Failed to generate insight: {str(e)}"
        )


# Optional: Get user's health data
@app.get("/api/v1/data/{user_id}")
async def get_user_data(
    user_id: str,
    limit: int = 50,
    api_key: str = Depends(verify_api_key)
):
    """Retrieve user's health data"""
    try:
        table = dynamodb.Table(DYNAMODB_TABLE)
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key('user_id').eq(user_id),
            Limit=limit,
            ScanIndexForward=False
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "data": response.get('Items', []),
            "count": len(response.get('Items', []))
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)