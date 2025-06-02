"""Gemini Health Insights API Endpoints.

This module provides REST API routes that expose the GeminiService functionality
to enable "chat with your health data" from frontend applications.

Endpoints include generating health insights, retrieving cached results,
and health status monitoring with proper Firebase authentication.
"""

from datetime import UTC, datetime
import logging
from typing import Any, Dict, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from clarity.auth import UserContext
from clarity.auth.firebase_auth import get_current_user
from clarity.core.interfaces import IAuthProvider, IConfigProvider
from clarity.ml.gemini_service import (
    GeminiService,
    HealthInsightRequest,
    HealthInsightResponse,
)

logger = logging.getLogger(__name__)

# Global dependencies - will be injected by container
_auth_provider: IAuthProvider | None = None
_config_provider: IConfigProvider | None = None
_gemini_service: GeminiService | None = None


def set_dependencies(
    auth_provider: IAuthProvider,
    config_provider: IConfigProvider,
) -> None:
    """Set dependencies for the router (called by container)."""
    global _auth_provider, _config_provider, _gemini_service
    _auth_provider = auth_provider
    _config_provider = config_provider

    # Initialize Gemini service
    if config_provider.is_development():
        logger.info("🧪 Gemini insights running in development mode")
        # In development, we might not have real Vertex AI credentials
        _gemini_service = GeminiService(project_id="dev-project")
    else:
        # Production setup
        gcp_project_id = config_provider.get_gcp_project_id()
        _gemini_service = GeminiService(project_id=gcp_project_id)


def get_gemini_service() -> GeminiService:
    """Get the Gemini service instance."""
    if _gemini_service is None:
        msg = "Gemini service not initialized"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=msg
        )
    return _gemini_service


# Request/Response Models
class InsightGenerationRequest(BaseModel):
    """Request for generating health insights."""

    analysis_results: dict[str, Any] = Field(
        description="PAT analysis results or health data metrics"
    )
    context: str | None = Field(
        None,
        description="Additional context for insights generation"
    )
    insight_type: str = Field(
        default="comprehensive",
        description="Type of insight to generate (comprehensive, brief, detailed)"
    )
    include_recommendations: bool = Field(
        default=True,
        description="Include actionable recommendations"
    )
    language: str = Field(
        default="en",
        description="Language code for insights"
    )


class InsightGenerationResponse(BaseModel):
    """Response for insight generation."""

    success: bool
    data: HealthInsightResponse
    metadata: dict[str, Any]


class InsightHistoryResponse(BaseModel):
    """Response for insight history."""

    success: bool
    data: dict[str, Any]
    metadata: dict[str, Any]


class ServiceStatusResponse(BaseModel):
    """Response for service status."""

    success: bool
    data: dict[str, Any]
    metadata: dict[str, Any]


# Error response models
class ErrorDetail(BaseModel):
    """Error detail structure."""

    code: str
    message: str
    details: dict[str, Any] | None = None
    request_id: str
    timestamp: str
    suggested_action: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: ErrorDetail


# Create router
router = APIRouter(prefix="/insights", tags=["gemini-insights"])


def generate_request_id() -> str:
    """Generate unique request ID."""
    return f"req_insights_{uuid.uuid4().hex[:8]}"


def create_metadata(request_id: str, processing_time_ms: float | None = None) -> dict[str, Any]:
    """Create standard metadata for responses."""
    metadata: dict[str, Any] = {
        "request_id": request_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "service": "gemini-insights",
        "version": "1.0.0"
    }

    if processing_time_ms is not None:
        metadata["processing_time_ms"] = processing_time_ms

    return metadata


def create_error_response(
    error_code: str,
    message: str,
    request_id: str,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    details: dict[str, Any] | None = None,
    suggested_action: str | None = None
) -> HTTPException:
    """Create standardized error response."""
    error_detail = ErrorDetail(
        code=error_code,
        message=message,
        details=details,
        request_id=request_id,
        timestamp=datetime.now(UTC).isoformat(),
        suggested_action=suggested_action
    )

    return HTTPException(
        status_code=status_code,
        detail=error_detail.model_dump()
    )


@router.post(
    "/generate",
    response_model=InsightGenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Health Insights",
    description="Generate AI-powered health insights from analysis results using Gemini 2.5 Pro"
)
async def generate_insights(
    request: Request,
    insight_request: InsightGenerationRequest,
    current_user: UserContext = Depends(get_current_user),
    gemini_service: GeminiService = Depends(get_gemini_service)
) -> InsightGenerationResponse:
    """Generate new health insights from analysis data.
    
    This endpoint uses the Gemini 2.5 Pro LLM to generate human-readable
    health insights and recommendations from structured analysis results.
    
    Args:
        insight_request: The insight generation request data
        current_user: Authenticated user information
        gemini_service: Gemini service instance
        
    Returns:
        Generated health insights with narrative and recommendations
        
    Raises:
        HTTPException: If insight generation fails or user lacks permissions
    """
    request_id = generate_request_id()
    start_time = datetime.now(UTC)

    try:
        logger.info(
            "🤖 Generating insights for user %s (request: %s)",
            current_user.user_id,
            request_id
        )

        # Check user permissions - for now, allow all authenticated users
        # In production, you would check specific permissions:
        # if Permission.READ_INSIGHTS not in current_user.permissions:
        if not current_user.is_active:
            raise create_error_response(
                error_code="ACCOUNT_DISABLED",
                message="User account is disabled",
                request_id=request_id,
                status_code=status.HTTP_403_FORBIDDEN,
                details={"user_id": current_user.user_id},
                suggested_action="contact_support"
            )

        # Create Gemini service request
        gemini_request = HealthInsightRequest(
            user_id=current_user.user_id,
            analysis_results=insight_request.analysis_results,
            context=insight_request.context,
            insight_type=insight_request.insight_type
        )

        # Generate insights
        logger.info("   • Calling Gemini service for insight generation...")
        insights = await gemini_service.generate_health_insights(gemini_request)

        # Calculate processing time
        processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

        logger.info(
            "✅ Insights generated successfully (%.1fms, confidence: %.2f)",
            processing_time,
            insights.confidence_score
        )

        # Create response
        return InsightGenerationResponse(
            success=True,
            data=insights,
            metadata=create_metadata(request_id, processing_time)
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.exception(
            "💥 Failed to generate insights for user %s (request: %s): %s",
            current_user.user_id,
            request_id,
            str(e)
        )

        raise create_error_response(
            error_code="INSIGHT_GENERATION_FAILED",
            message="Failed to generate health insights",
            request_id=request_id,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error_type": type(e).__name__, "error_message": str(e)},
            suggested_action="retry_later"
        )


@router.get(
    "/{insight_id}",
    response_model=InsightGenerationResponse,
    summary="Get Cached Insight",
    description="Retrieve a previously generated insight by ID"
)
async def get_insight(
    insight_id: str,
    current_user: UserContext = Depends(get_current_user)
) -> InsightGenerationResponse:
    """Retrieve cached insights by ID.
    
    Args:
        insight_id: The ID of the insight to retrieve
        current_user: Authenticated user information
        
    Returns:
        The cached insight data
        
    Raises:
        HTTPException: If insight not found or access denied
    """
    request_id = generate_request_id()

    logger.info(
        "📄 Retrieving insight %s for user %s (request: %s)",
        insight_id,
        current_user.user_id,
        request_id
    )

    # For now, return a not implemented response
    # In a full implementation, this would query Firestore or cache
    raise create_error_response(
        error_code="NOT_IMPLEMENTED",
        message="Insight retrieval not yet implemented",
        request_id=request_id,
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        suggested_action="use_generate_endpoint"
    )


@router.get(
    "/history/{user_id}",
    response_model=InsightHistoryResponse,
    summary="Get Insight History",
    description="Retrieve insight generation history for a user"
)
async def get_insight_history(
    user_id: str,
    limit: int = 10,
    offset: int = 0,
    current_user: UserContext = Depends(get_current_user)
) -> InsightHistoryResponse:
    """Get insight history for a user.
    
    Args:
        user_id: The user ID to get history for
        limit: Maximum number of insights to return
        offset: Number of insights to skip
        current_user: Authenticated user information
        
    Returns:
        List of historical insights
        
    Raises:
        HTTPException: If access denied or user not found
    """
    request_id = generate_request_id()

    # Check if user is requesting their own data or has admin permissions
    # For now, only allow users to access their own data
    if current_user.user_id != user_id:
        raise create_error_response(
            error_code="ACCESS_DENIED",
            message="Cannot access another user's insight history",
            request_id=request_id,
            status_code=status.HTTP_403_FORBIDDEN,
            suggested_action="request_own_data"
        )

    logger.info(
        "📚 Retrieving insight history for user %s (request: %s)",
        user_id,
        request_id
    )

    # For now, return a not implemented response
    # In a full implementation, this would query Firestore
    raise create_error_response(
        error_code="NOT_IMPLEMENTED",
        message="Insight history not yet implemented",
        request_id=request_id,
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        suggested_action="use_generate_endpoint"
    )


@router.get(
    "/status",
    response_model=ServiceStatusResponse,
    summary="Service Health Status",
    description="Check the health status of the Gemini insights service"
)
async def get_service_status(
    gemini_service: GeminiService = Depends(get_gemini_service)
) -> ServiceStatusResponse:
    """Check Gemini service health status.
    
    Args:
        gemini_service: Gemini service instance
        
    Returns:
        Service health status and metrics
    """
    request_id = generate_request_id()
    start_time = datetime.now(UTC)

    try:
        logger.info("🏥 Checking Gemini service health (request: %s)", request_id)

        # Get service health
        health_status = await gemini_service.health_check()

        # Calculate response time
        processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

        # Create comprehensive status response
        status_data = {
            "service_name": "gemini-insights",
            "status": "healthy" if health_status["initialized"] else "degraded",
            "gemini_service": health_status,
            "capabilities": {
                "insight_generation": True,
                "narrative_creation": True,
                "health_recommendations": True,
                "multi_language": False  # Not yet implemented
            },
            "performance_metrics": {
                "health_check_time_ms": processing_time,
                "average_insight_generation_time": "15-30 seconds",
                "supported_languages": ["en"]
            },
            "version_info": {
                "api_version": "1.0.0",
                "gemini_model": "gemini-2.5-pro",
                "service_version": "1.0.0"
            }
        }

        logger.info(
            "✅ Service status check completed (%.1fms) - Status: %s",
            processing_time,
            status_data["status"]
        )

        return ServiceStatusResponse(
            success=True,
            data=status_data,
            metadata=create_metadata(request_id, processing_time)
        )

    except Exception as e:
        logger.exception(
            "💥 Service status check failed (request: %s): %s",
            request_id,
            str(e)
        )

        # Return degraded status instead of error
        status_data = {
            "service_name": "gemini-insights",
            "status": "degraded",
            "error": str(e),
            "last_error_time": datetime.now(UTC).isoformat()
        }

        return ServiceStatusResponse(
            success=False,
            data=status_data,
            metadata=create_metadata(request_id)
        )


# Export router
__all__ = ["router", "set_dependencies"]
