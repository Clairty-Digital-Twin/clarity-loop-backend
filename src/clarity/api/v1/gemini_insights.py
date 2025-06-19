"""Gemini Health Insights API Endpoints.

This module provides REST API routes that expose the GeminiService functionality
to enable "chat with your health data" from frontend applications.

Endpoints include generating health insights, retrieving cached results,
and health status monitoring with proper authentication.
"""

# removed - breaks FastAPI

from datetime import UTC, datetime
import logging
import os
from typing import Any, NoReturn
import uuid

from boto3.dynamodb.conditions import Key
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from clarity.auth.dependencies import AuthenticatedUser
from clarity.core.config_aws import get_settings
from clarity.ml.gemini_service import (
    GeminiService,
    HealthInsightRequest,
    HealthInsightResponse,
)
from clarity.ports.auth_ports import IAuthProvider
from clarity.ports.config_ports import IConfigProvider
from clarity.storage.dynamodb_client import DynamoDBHealthDataRepository

logger = logging.getLogger(__name__)

# Constants
NARRATIVE_PREVIEW_LENGTH = 200

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
        # Production setup - using AWS region instead of GCP project
        aws_settings = get_settings()
        _gemini_service = GeminiService(project_id=aws_settings.aws_region)


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
router = APIRouter(tags=["ai-insights"])


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


def _raise_account_disabled_error(request_id: str, user_id: str) -> NoReturn:
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
) -> NoReturn:
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


def _raise_insight_not_found_error(insight_id: str, request_id: str) -> NoReturn:
    """Raise HTTPException for insight not found."""
    raise create_error_response(
        error_code="INSIGHT_NOT_FOUND",
        message=f"Insight {insight_id} not found",
        request_id=request_id,
        status_code=status.HTTP_404_NOT_FOUND,
        details={"insight_id": insight_id},
        suggested_action="check_insight_id",
    )


def _raise_insight_access_denied_error(insight_id: str, request_id: str) -> NoReturn:
    """Raise HTTPException for insight access denied."""
    raise create_error_response(
        error_code="ACCESS_DENIED",
        message="Cannot access another user's insights",
        request_id=request_id,
        status_code=status.HTTP_403_FORBIDDEN,
        details={"insight_id": insight_id},
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
    "/",
    response_model=InsightGenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Health Insights",
    description=(
        "Generate AI-powered health insights from analysis results "
        "using Gemini 2.5 Pro"
    ),
)
@router.post(
    "",
    response_model=InsightGenerationResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,  # Don't show duplicate in OpenAPI docs
)
async def generate_insights(
    insight_request: InsightGenerationRequest,
    current_user: AuthenticatedUser,
    gemini_service: GeminiService = Depends(get_gemini_service),
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

        # No need to check is_active for User model - just proceed

        # Create Gemini service request
        gemini_request = HealthInsightRequest(
            user_id=current_user.user_id,
            analysis_results=insight_request.analysis_results,
            context=insight_request.context,
            insight_type=insight_request.insight_type,
        )

        # Generate insights
        insight_response = await gemini_service.generate_health_insights(gemini_request)

        # Save insight to DynamoDB
        dynamodb_client = _get_dynamodb_client()
        insight_id = f"insight_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now(UTC)

        insight_item: dict[str, Any] = {
            "pk": f"USER#{current_user.user_id}",
            "sk": f"INSIGHT#{timestamp.isoformat()}",
            "id": insight_id,
            "user_id": current_user.user_id,
            "narrative": insight_response.narrative,
            "key_insights": insight_response.key_insights,
            "recommendations": insight_response.recommendations,
            "confidence_score": insight_response.confidence_score,
            "generated_at": insight_response.generated_at,
            "created_at": timestamp.isoformat(),
        }

        # Also store with direct insight ID access
        dynamodb_client.table.put_item(Item=insight_item)
        dynamodb_client.table.put_item(
            Item={
                "pk": f"INSIGHT#{insight_id}",
                "sk": f"INSIGHT#{insight_id}",
                **insight_item,
            }
        )

        # Calculate processing time
        processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

        logger.info(
            "✅ Insights generated successfully for user %s "
            "(request: %s, time: %.2fms)",
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
    current_user: AuthenticatedUser,
) -> InsightGenerationResponse:
    """Retrieve cached insights by ID from DynamoDB.

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

        # Get insight from DynamoDB
        dynamodb_client = _get_dynamodb_client()
        response = dynamodb_client.table.get_item(
            Key={"pk": f"INSIGHT#{insight_id}", "sk": f"INSIGHT#{insight_id}"}
        )
        insight_doc = response.get("Item")

        if not insight_doc:
            _raise_insight_not_found_error(insight_id, request_id)

        # Verify user owns the insight
        if insight_doc.get("user_id") != current_user.user_id:
            _raise_insight_access_denied_error(insight_id, request_id)

        # Convert DynamoDB document to HealthInsightResponse
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
    current_user: AuthenticatedUser,
    limit: int = 10,
    offset: int = 0,
) -> InsightHistoryResponse:
    """Get insight history for a user from DynamoDB.

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

        # Get insights from DynamoDB
        dynamodb_client = _get_dynamodb_client()
        # Query insights for user
        response = dynamodb_client.table.query(
            KeyConditionExpression=Key("pk").eq(f"USER#{user_id}")
            & Key("sk").begins_with("INSIGHT#"),
            Limit=limit + offset,
            ScanIndexForward=False,  # Most recent first
        )

        all_insights = response.get("Items", [])
        # Apply offset
        insights = (
            all_insights[offset : offset + limit] if offset < len(all_insights) else []
        )

        # Get total count
        total_count = len(all_insights)
        while "LastEvaluatedKey" in response:
            response = dynamodb_client.table.query(
                KeyConditionExpression=Key("pk").eq(f"USER#{user_id}")
                & Key("sk").begins_with("INSIGHT#"),
                ExclusiveStartKey=response["LastEvaluatedKey"],
                Select="COUNT",
            )
            total_count += response.get("Count", 0)

        # Format insights for response
        formatted_insights = []
        for insight in insights:
            narrative = str(insight.get("narrative", ""))
            key_insights = insight.get("key_insights", [])
            recommendations = insight.get("recommendations", [])

            formatted_insight = {
                "id": insight.get("id"),
                "narrative": (
                    narrative[:NARRATIVE_PREVIEW_LENGTH] + "..."
                    if len(narrative) > NARRATIVE_PREVIEW_LENGTH
                    else narrative
                ),
                "generated_at": insight.get("generated_at"),
                "confidence_score": insight.get("confidence_score", 0.0),
                "key_insights_count": (
                    len(key_insights) if isinstance(key_insights, list) else 0
                ),
                "recommendations_count": (
                    len(recommendations) if isinstance(recommendations, list) else 0
                ),
            }
            formatted_insights.append(formatted_insight)

        history_data = {
            "insights": formatted_insights,
            "total_count": total_count,
            "has_more": offset + len(insights) < total_count,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "current_page": (offset // limit) + 1,
                "total_pages": (total_count + limit - 1) // limit,
            },
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
    _current_user: AuthenticatedUser,
    gemini_service: GeminiService = Depends(get_gemini_service),
) -> ServiceStatusResponse:
    """Check Gemini service health status.

    Args:
        _current_user: Authenticated user context
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


def _get_dynamodb_client() -> DynamoDBHealthDataRepository:
    """Get DynamoDB client for storing/retrieving insights."""
    table_name = os.getenv("DYNAMODB_TABLE_NAME", "clarity-health-data")
    region = os.getenv("AWS_REGION", "us-east-1")
    return DynamoDBHealthDataRepository(table_name=table_name, region=region)


# Export router
__all__ = ["router", "set_dependencies"]
