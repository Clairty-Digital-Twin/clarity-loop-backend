"""Gemini Health Insights API Endpoints.

This module provides REST API routes that expose the GeminiService functionality
to enable "chat with your health data" from frontend applications.

Endpoints include generating health insights, retrieving cached results,
and health status monitoring with proper Firebase authentication.
"""

from datetime import UTC, datetime
import logging
from typing import Any
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from clarity.auth import UserContext
from clarity.auth.firebase_auth import get_current_user
from clarity.core.user_context import UserContext
from clarity.ports.auth_ports import IAuthProvider
from clarity.ports.config_ports import IConfigProvider
from clarity.services.gemini_service import (
    GeminiService,
    HealthInsightRequest,
    HealthInsightResponse,
)
from clarity.storage.firestore_client import FirestoreClient

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
    # Note: Using globals here for FastAPI dependency injection pattern
    # This is the recommended approach for this architecture
    global _auth_provider, _config_provider, _gemini_service  # noqa: PLW0603
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg
        )
    return _gemini_service


# Request/Response Models
class InsightGenerationRequest(BaseModel):
    """Request for generating health insights."""

    analysis_results: dict[str, Any] = Field(
        description="PAT analysis results or health data metrics"
    )
    context: str | None = Field(
        None, description="Additional context for insights generation"
    )
    insight_type: str = Field(
        default="comprehensive",
        description="Type of insight to generate (comprehensive, brief, detailed)",
    )
    include_recommendations: bool = Field(
        default=True, description="Include actionable recommendations"
    )
    language: str = Field(default="en", description="Language code for insights")


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


def create_metadata(
    request_id: str, processing_time_ms: float | None = None
) -> dict[str, Any]:
    """Create standard metadata for responses."""
    metadata: dict[str, Any] = {
        "request_id": request_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "service": "gemini-insights",
        "version": "1.0.0",
    }

    if processing_time_ms is not None:
        metadata["processing_time_ms"] = processing_time_ms

    return metadata


def _raise_account_disabled_error(request_id: str, user_id: str) -> None:
    """Raise account disabled error."""
    raise create_error_response(
        error_code="ACCOUNT_DISABLED",
        message="User account is disabled",
        request_id=request_id,
        status_code=status.HTTP_403_FORBIDDEN,
        details={"user_id": user_id},
        suggested_action="contact_support",
    )


def _raise_access_denied_error(
    user_id: str, current_user_id: str, request_id: str
) -> None:
    """Raise access denied error for insight history."""
    raise create_error_response(
        error_code="ACCESS_DENIED",
        message="Cannot access another user's insight history",
        request_id=request_id,
        status_code=status.HTTP_403_FORBIDDEN,
        details={
            "requested_user_id": user_id,
            "current_user_id": current_user_id,
        },
        suggested_action="check_permissions",
    )


def create_error_response(
    error_code: str,
    message: str,
    request_id: str,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    details: dict[str, Any] | None = None,
    suggested_action: str | None = None,
) -> HTTPException:
    """Create standardized error response."""
    error_detail = ErrorDetail(
        code=error_code,
        message=message,
        details=details,
        request_id=request_id,
        timestamp=datetime.now(UTC).isoformat(),
        suggested_action=suggested_action,
    )

    return HTTPException(status_code=status_code, detail=error_detail.model_dump())


@router.post(
    "/generate",
    response_model=InsightGenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Health Insights",
    description="Generate AI-powered health insights from analysis results using Gemini 2.5 Pro",
)
async def generate_insights(
    insight_request: InsightGenerationRequest,
    current_user: UserContext = Depends(get_current_user),  # noqa: B008
    gemini_service: GeminiService = Depends(get_gemini_service),  # noqa: B008
) -> InsightGenerationResponse:
    """Generate new health insights from analysis data.

    This endpoint uses the Gemini 2.5 Pro LLM to generate human-readable
    health insights and recommendations from structured analysis results.

    Args:
        insight_request: The insight generation request data
        current_user: Authenticated user context
        gemini_service: Gemini service instance

    Returns:
        InsightGenerationResponse: Generated insights with metadata

    Raises:
        HTTPException: If user is inactive or insight generation fails
    """
    request_id = generate_request_id()
    start_time = datetime.now(UTC)

    try:
        logger.info(
            "🔮 Generating insights for user %s (request: %s)",
            current_user.user_id,
            request_id,
        )

        # Validate user permissions
        # if Permission.READ_INSIGHTS not in current_user.permissions:
        if not current_user.is_active:
            _raise_account_disabled_error(request_id, current_user.user_id)

        # Create Gemini service request
        gemini_request = HealthInsightRequest(
            user_id=current_user.user_id,
            analysis_results=insight_request.analysis_results,
            context=insight_request.context,
            insight_type=insight_request.insight_type,
        )

        # Generate insights
        insight_response = await gemini_service.generate_health_insights(gemini_request)

        # Calculate processing time
        processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

        logger.info(
            "✅ Insights generated successfully for user %s (request: %s, time: %.2fms)",
            current_user.user_id,
            request_id,
            processing_time,
        )

        return InsightGenerationResponse(
            success=True,
            data=insight_response,
            metadata=create_metadata(request_id, processing_time),
        )

    except Exception as e:
        processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
        logger.exception(
            "💥 Insight generation failed for user %s (request: %s, time: %.2fms)",
            current_user.user_id,
            request_id,
            processing_time,
        )

        raise create_error_response(
            error_code="INSIGHT_GENERATION_FAILED",
            message="Failed to generate health insights",
            request_id=request_id,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error_type": type(e).__name__, "error_message": str(e)},
            suggested_action="retry_later",
        ) from e


@router.get(
    "/{insight_id}",
    response_model=InsightGenerationResponse,
    summary="Get Cached Insight",
    description="Retrieve a previously generated insight by ID",
)
async def get_insight(
    insight_id: str,
    current_user: UserContext = Depends(get_current_user),  # noqa: B008
) -> InsightGenerationResponse:
    """🔥 FIXED: Retrieve cached insights by ID from Firestore.

    Args:
        insight_id: Unique identifier for the insight
        current_user: Authenticated user context

    Returns:
        InsightGenerationResponse: Cached insight data

    Raises:
        HTTPException: If insight not found or access denied
    """
    request_id = generate_request_id()

    try:
        logger.info(
            "📖 Retrieving insight %s for user %s (request: %s)",
            insight_id,
            current_user.user_id,
            request_id,
        )

        # Get insight from Firestore
        firestore_client = _get_firestore_client()
        insight_doc = await firestore_client.get_document(
            collection="insights",
            document_id=insight_id
        )

        if not insight_doc:
            raise create_error_response(
                error_code="INSIGHT_NOT_FOUND",
                message=f"Insight {insight_id} not found",
                request_id=request_id,
                status_code=status.HTTP_404_NOT_FOUND,
                details={"insight_id": insight_id},
                suggested_action="check_insight_id",
            )

        # Verify user owns the insight
        if insight_doc.get("user_id") != current_user.user_id:
            raise create_error_response(
                error_code="ACCESS_DENIED",
                message="Cannot access another user's insights",
                request_id=request_id,
                status_code=status.HTTP_403_FORBIDDEN,
                details={"insight_id": insight_id},
                suggested_action="check_permissions",
            )

        # Convert Firestore document to HealthInsightResponse
        insight_response = HealthInsightResponse(
            user_id=insight_doc["user_id"],
            narrative=insight_doc.get("narrative", ""),
            key_insights=insight_doc.get("key_insights", []),
            recommendations=insight_doc.get("recommendations", []),
            confidence_score=insight_doc.get("confidence_score", 0.0),
            generated_at=insight_doc.get("generated_at", datetime.now(UTC).isoformat()),
        )

        return InsightGenerationResponse(
            success=True,
            data=insight_response,
            metadata=create_metadata(request_id),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "💥 Failed to retrieve insight %s for user %s (request: %s)",
            insight_id,
            current_user.user_id,
            request_id,
        )

        raise create_error_response(
            error_code="INSIGHT_RETRIEVAL_FAILED",
            message=f"Failed to retrieve insight {insight_id}",
            request_id=request_id,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"insight_id": insight_id, "error_message": str(e)},
            suggested_action="retry_later",
        ) from e


@router.get(
    "/history/{user_id}",
    response_model=InsightHistoryResponse,
    summary="Get Insight History",
    description="Retrieve insight generation history for a user",
)
async def get_insight_history(
    user_id: str,
    limit: int = 10,
    offset: int = 0,
    current_user: UserContext = Depends(get_current_user),  # noqa: B008
) -> InsightHistoryResponse:
    """🔥 FIXED: Get insight history for a user from Firestore.

    Args:
        user_id: User ID to get history for
        limit: Maximum number of insights to return
        offset: Number of insights to skip
        current_user: Authenticated user context

    Returns:
        InsightHistoryResponse: User's insight history

    Raises:
        HTTPException: If access denied or retrieval fails
    """
    request_id = generate_request_id()

    try:
        logger.info(
            "📚 Retrieving insight history for user %s (request: %s)",
            user_id,
            request_id,
        )

        # Validate user can access this history
        if current_user.user_id != user_id:
            _raise_access_denied_error(user_id, current_user.user_id, request_id)

        # Get insights from Firestore
        firestore_client = _get_firestore_client()
        insights = await firestore_client.query_documents(
            collection="insights",
            filters=[{"field": "user_id", "op": "==", "value": user_id}],
            limit=limit,
            offset=offset,
            order_by="generated_at",
            order_direction="desc"
        )

        # Get total count for pagination
        total_count = await firestore_client.count_documents(
            collection="insights",
            filters=[{"field": "user_id", "op": "==", "value": user_id}]
        )

        # Format insights for response
        formatted_insights = [{
                "id": insight.get("id"),
                "narrative": insight.get("narrative", "")[:200] + "..." if len(insight.get("narrative", "")) > 200 else insight.get("narrative", ""),
                "generated_at": insight.get("generated_at"),
                "confidence_score": insight.get("confidence_score", 0.0),
                "key_insights_count": len(insight.get("key_insights", [])),
                "recommendations_count": len(insight.get("recommendations", []))
            } for insight in insights]

        history_data = {
            "insights": formatted_insights,
            "total_count": total_count,
            "has_more": offset + len(insights) < total_count,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "current_page": (offset // limit) + 1,
                "total_pages": (total_count + limit - 1) // limit
            }
        }

        return InsightHistoryResponse(
            success=True, data=history_data, metadata=create_metadata(request_id)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "💥 Failed to retrieve insight history for user %s (request: %s)",
            user_id,
            request_id,
        )

        raise create_error_response(
            error_code="HISTORY_RETRIEVAL_FAILED",
            message="Failed to retrieve insight history",
            request_id=request_id,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"user_id": user_id, "error_message": str(e)},
            suggested_action="retry_later",
        ) from e


@router.get(
    "/status",
    response_model=ServiceStatusResponse,
    summary="Service Health Status",
    description="Check the health status of the Gemini insights service",
)
async def get_service_status(
    gemini_service: GeminiService = Depends(get_gemini_service),  # noqa: B008
) -> ServiceStatusResponse:
    """Check Gemini service health status.

    Args:
        gemini_service: Gemini service instance

    Returns:
        ServiceStatusResponse: Service health status

    Raises:
        HTTPException: If status check fails
    """
    request_id = generate_request_id()

    try:
        logger.info("🔍 Checking Gemini service status (request: %s)", request_id)

        # Check service health
        is_healthy = gemini_service.is_initialized
        model_info = {
            "model_name": "gemini-2.0-flash-exp",
            "project_id": gemini_service.project_id,
            "initialized": is_healthy,
            "capabilities": [
                "health_insights_generation",
                "contextual_analysis",
                "recommendation_generation",
            ],
        }

        status_data = {
            "service": "gemini-insights",
            "status": "healthy" if is_healthy else "unhealthy",
            "model": model_info,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        logger.info(
            "✅ Service status check completed (request: %s, status: %s)",
            request_id,
            status_data["status"],
        )

        return ServiceStatusResponse(
            success=True, data=status_data, metadata=create_metadata(request_id)
        )

    except Exception as e:
        logger.exception("💥 Service status check failed (request: %s)", request_id)

        raise create_error_response(
            error_code="STATUS_CHECK_FAILED",
            message="Failed to check service status",
            request_id=request_id,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error_type": type(e).__name__, "error_message": str(e)},
            suggested_action="check_service_health",
        ) from e


def _get_firestore_client() -> FirestoreClient:
    """Get Firestore client for storing/retrieving insights."""
    import os

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "clarity-digital-twin")
    return FirestoreClient(project_id=project_id)


# Export router
__all__ = ["router", "set_dependencies"]
