"""FastAPI endpoints for PAT (Pretrained Actigraphy Transformer) Analysis Service.

This module provides REST API endpoints for actigraphy analysis using the PAT model,
enabling the core "chat with your actigraphy" vertical slice functionality.

Endpoints:
- POST /analyze - Submit actigraphy data for PAT analysis
- GET /analysis/{analysis_id} - Retrieve analysis results
- POST /analyze-step-data - Submit Apple HealthKit step data for analysis
- GET /health - Service health check
"""

# removed - breaks FastAPI

from datetime import UTC, datetime
import logging
import os
from typing import Any, cast
import uuid

from boto3.dynamodb.conditions import Key
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, validator
from slowapi import Limiter

from clarity.auth.dependencies import AuthenticatedUser
from clarity.middleware.rate_limiting import get_user_id_or_ip
from clarity.ml.inference_engine import AsyncInferenceEngine, get_inference_engine
from clarity.ml.pat_service import ActigraphyAnalysis, ActigraphyInput
from clarity.ml.preprocessing import ActigraphyDataPoint
from clarity.ml.proxy_actigraphy import (
    StepCountData,
    create_proxy_actigraphy_transformer,
)
from clarity.storage.dynamodb_client import DynamoDBHealthDataRepository

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(tags=["pat-analysis"])

# Create rate limiter for AI endpoints (more restrictive due to resource intensity)
ai_limiter = Limiter(key_func=get_user_id_or_ip)


# 🔥 FIXED: Response model for PAT analysis results - moved before usage
class PATAnalysisResponse(BaseModel):
    """Response model for PAT analysis results."""

    processing_id: str = Field(description="Processing ID for the analysis")
    status: str = Field(description="Status: completed, processing, failed, not_found")
    message: str | None = Field(None, description="Status message")
    analysis_date: str | None = Field(None, description="When analysis was completed")
    pat_features: dict[str, float] | None = Field(
        None, description="PAT model features"
    )
    activity_embedding: list[float] | None = Field(
        None, description="Activity embedding vector"
    )
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class StepDataRequest(BaseModel):
    """Request for Apple HealthKit step data analysis."""

    step_counts: list[int] = Field(
        description="Minute-by-minute step counts (10,080 values for 1 week)",
        min_length=1,
        max_length=20160,  # 2 weeks max
    )
    timestamps: list[datetime] = Field(
        description="Corresponding timestamps for each step count"
    )
    user_metadata: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Optional user demographics (age_group, sex, etc.) - limited size",
        max_length=50,  # Prevent oversized metadata dictionaries
    )

    @validator("timestamps")
    @classmethod
    def validate_timestamps_match_steps(
        cls, v: list[datetime], values: dict[str, Any]
    ) -> list[datetime]:
        """Ensure timestamps match step count length."""
        if "step_counts" in values and len(v) != len(values["step_counts"]):
            msg = "Timestamps length must match step_counts length"
            raise ValueError(msg)
        return v


class DirectActigraphyRequest(BaseModel):
    """Request for direct actigraphy data analysis."""

    data_points: list[dict[str, Any]] = Field(
        description="Actigraphy data points with timestamp and value"
    )
    sampling_rate: float = Field(default=1.0, description="Samples per minute", gt=0)
    duration_hours: int = Field(
        default=24,
        description="Duration in hours",
        ge=1,
        le=168,  # 1 week max
    )


class AnalysisResponse(BaseModel):
    """Response for analysis requests."""

    analysis_id: str = Field(description="Unique analysis identifier")
    status: str = Field(description="Analysis status: processing, completed, failed")
    analysis: ActigraphyAnalysis | None = Field(
        default=None, description="Analysis results (available when status=completed)"
    )
    processing_time_ms: float | None = Field(
        default=None, description="Processing time in milliseconds"
    )
    message: str | None = Field(
        default=None, description="Status message or error description"
    )
    cached: bool = Field(
        default=False, description="Whether result was retrieved from cache"
    )


class HealthCheckResponse(BaseModel):
    """PAT service health check response."""

    service: str
    status: str
    timestamp: str
    version: str = "1.0.0"
    inference_engine: dict[str, Any]
    pat_model: dict[str, Any]


async def get_pat_inference_engine() -> AsyncInferenceEngine:
    """Dependency to get the PAT inference engine."""
    return await get_inference_engine()


@router.post(
    "/step-analysis",
    response_model=AnalysisResponse,
    summary="Analyze Apple HealthKit Step Data",
    description=(
        "Submit Apple HealthKit step count data for PAT analysis "
        "using proxy actigraphy transformation"
    ),
)
@router.post(
    "/step-analysis/",
    response_model=AnalysisResponse,
    include_in_schema=False,  # Don't show duplicate in OpenAPI docs
)
@ai_limiter.limit("20/hour")  # AI endpoints are resource-intensive
async def analyze_step_data(
    request: Request,
    step_request: StepDataRequest,
    _background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser,
    inference_engine: AsyncInferenceEngine = Depends(get_pat_inference_engine),
) -> AnalysisResponse:
    """Analyze Apple HealthKit step data using PAT model with proxy actigraphy transformation.

    This endpoint is the core of the "chat with your actigraphy" vertical slice,
    enabling analysis of Apple HealthKit step data through:
    1. Proxy actigraphy transformation
    2. PAT model inference
    3. Clinical insight generation
    """
    _ = request  # Used by rate limiter
    analysis_id = str(uuid.uuid4())

    try:
        logger.info(
            "Starting step data analysis for user %s, analysis_id %s",
            current_user.user_id,
            analysis_id,
        )

        # Transform step data to proxy actigraphy
        transformer = create_proxy_actigraphy_transformer()

        step_data = StepCountData(
            user_id=current_user.user_id,
            upload_id=analysis_id,
            step_counts=[
                float(count) for count in step_request.step_counts
            ],  # Convert to float
            timestamps=step_request.timestamps,
        )

        # Generate proxy actigraphy vector
        proxy_result = transformer.transform_step_data(step_data)
        logger.info(
            "Proxy actigraphy transformation complete: quality=%.3f",
            proxy_result.quality_score,
        )

        # Convert to ActigraphyDataPoint format for PAT model
        actigraphy_points = [
            ActigraphyDataPoint(
                timestamp=step_request.timestamps[i],
                value=float(proxy_result.vector[i]),
            )
            for i in range(len(proxy_result.vector))
        ]

        # Create PAT input
        pat_input = ActigraphyInput(
            user_id=current_user.user_id,
            data_points=actigraphy_points,
            sampling_rate=1.0,  # 1 sample per minute
            duration_hours=len(proxy_result.vector) // 60,
        )

        # Submit for async inference
        inference_response = await inference_engine.predict(
            input_data=pat_input,
            request_id=analysis_id,
            timeout_seconds=60.0,
            cache_enabled=True,
        )

        logger.info(
            "PAT analysis complete for %s, cached=%s",
            analysis_id,
            inference_response.cached,
        )

        return AnalysisResponse(
            analysis_id=analysis_id,
            status="completed",
            analysis=inference_response.analysis,
            processing_time_ms=inference_response.processing_time_ms,
            message="Analysis completed successfully",
            cached=inference_response.cached,
        )

    except Exception as e:
        logger.exception("Step data analysis failed for %s", analysis_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "analysis_id": analysis_id,
                "error": "Analysis failed",
                "message": str(e),
            },
        ) from e


@router.post(
    "/analysis",
    response_model=AnalysisResponse,
    summary="Analyze Direct Actigraphy Data",
    description="Submit preprocessed actigraphy data for PAT analysis",
)
@router.post(
    "/analysis/",
    response_model=AnalysisResponse,
    include_in_schema=False,  # Don't show duplicate in OpenAPI docs
)
async def analyze_actigraphy_data(
    request: DirectActigraphyRequest,
    current_user: AuthenticatedUser,
    inference_engine: AsyncInferenceEngine = Depends(get_pat_inference_engine),
) -> AnalysisResponse:
    """Analyze preprocessed actigraphy data using the PAT model.

    This endpoint accepts direct actigraphy data that has already been preprocessed
    and formatted for PAT model analysis.
    """
    analysis_id = str(uuid.uuid4())

    try:
        logger.info(
            "Starting direct actigraphy analysis for user %s, analysis_id %s",
            current_user.user_id,
            analysis_id,
        )

        # Convert request data to ActigraphyDataPoint format
        actigraphy_points = [
            ActigraphyDataPoint(
                timestamp=(
                    datetime.fromisoformat(point["timestamp"])
                    if isinstance(point["timestamp"], str)
                    else point["timestamp"]
                ),
                value=float(point["value"]),
            )
            for point in request.data_points
        ]

        # Create PAT input
        pat_input = ActigraphyInput(
            user_id=current_user.user_id,
            data_points=actigraphy_points,
            sampling_rate=request.sampling_rate,
            duration_hours=request.duration_hours,
        )

        # Submit for async inference
        inference_response = await inference_engine.predict(
            input_data=pat_input,
            request_id=analysis_id,
            timeout_seconds=60.0,
            cache_enabled=True,
        )

        logger.info(
            "Direct actigraphy analysis complete for %s, cached=%s",
            analysis_id,
            inference_response.cached,
        )

        return AnalysisResponse(
            analysis_id=analysis_id,
            status="completed",
            analysis=inference_response.analysis,
            processing_time_ms=inference_response.processing_time_ms,
            message="Analysis completed successfully",
            cached=inference_response.cached,
        )

    except Exception as e:
        logger.exception("Direct actigraphy analysis failed for %s", analysis_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "analysis_id": analysis_id,
                "error": "Analysis failed",
                "message": str(e),
            },
        ) from e


@router.get(
    "/analysis/{processing_id}",
    summary="Get PAT Analysis Results",
    description="""
    Retrieve PAT (Pretrained Actigraphy Transformer) analysis results for a specific processing ID.

    **Returns:**
    - PAT model embeddings and predictions
    - Activity pattern analysis
    - Sleep quality metrics
    - Circadian rhythm insights
    - Processing metadata and status

    **Status Values:**
    - `completed`: Analysis finished successfully
    - `processing`: Analysis still running
    - `failed`: Analysis encountered an error
    - `not_found`: No analysis found for this ID
    """,
    response_model=PATAnalysisResponse,
    responses={
        200: {"description": "Analysis results retrieved successfully"},
        404: {"description": "Analysis not found"},
        503: {"description": "Service temporarily unavailable"},
    },
)
async def get_pat_analysis(
    processing_id: str,
    current_user: AuthenticatedUser,
) -> PATAnalysisResponse:
    """Retrieve actual PAT analysis results from DynamoDB."""
    try:
        logger.info(
            "Retrieving PAT analysis results: %s for user %s",
            processing_id,
            current_user.user_id,
        )

        # Get DynamoDB client
        dynamodb_client = _get_analysis_repository()

        # Try to get analysis results from DynamoDB
        # Query for analysis results
        response = dynamodb_client.table.query(
            KeyConditionExpression=Key("pk").eq(f"USER#{current_user.user_id}")
            & Key("sk").eq(f"ANALYSIS#{processing_id}")
        )

        items = response.get("Items", [])
        analysis_result = items[0] if items else None

        if analysis_result:
            # Found completed analysis results
            logger.info("Successfully retrieved PAT analysis: %s", processing_id)
            return PATAnalysisResponse(
                processing_id=processing_id,
                status=analysis_result.get("status", "completed"),
                message="Analysis completed successfully",
                analysis_date=analysis_result.get("analysis_date"),
                pat_features=analysis_result.get("pat_features", {}),
                activity_embedding=analysis_result.get("activity_embedding", []),
                metadata={
                    "cardio_features": analysis_result.get("cardio_features", []),
                    "respiratory_features": analysis_result.get(
                        "respiratory_features", []
                    ),
                    "activity_features": analysis_result.get("activity_features", []),
                    "fused_vector": analysis_result.get("fused_vector", []),
                    "summary_stats": analysis_result.get("summary_stats", {}),
                    "processing_metadata": analysis_result.get(
                        "processing_metadata", {}
                    ),
                },
            )

        # If not found in analysis_results, check processing_jobs in DynamoDB
        job_response = dynamodb_client.table.get_item(
            Key={"pk": f"JOB#{processing_id}", "sk": f"JOB#{processing_id}"}
        )
        processing_status = job_response.get("Item")

        if processing_status:
            # Cast to dict since we know it's not None
            processing_status_dict = cast("dict[str, Any]", processing_status)

            # Check if user owns this processing job
            if processing_status_dict.get("user_id") != current_user.user_id:
                logger.warning(
                    "User %s attempted to access processing job %s",
                    current_user.user_id,
                    processing_id,
                )
                return PATAnalysisResponse(
                    processing_id=processing_id,
                    status="not_found",
                    message="Analysis not found",
                    analysis_date=None,
                    pat_features=None,
                    activity_embedding=None,
                    metadata={},
                )

            # Return status from processing job
            job_status = processing_status_dict.get("status", "unknown")
            return PATAnalysisResponse(
                processing_id=processing_id,
                status=job_status,
                message=f"Analysis is {job_status}. Results will be available when processing completes.",
                analysis_date=processing_status_dict.get("created_at"),
                pat_features=None,
                activity_embedding=None,
                metadata=processing_status_dict,
            )

        # Neither analysis results nor processing job found
        logger.warning("PAT analysis not found: %s", processing_id)
        return PATAnalysisResponse(
            processing_id=processing_id,
            status="not_found",
            message="Analysis results not found. Processing may never have started or may have been removed.",
            analysis_date=None,
            pat_features=None,
            activity_embedding=None,
            metadata={},
        )

    except Exception as e:
        logger.exception("Error retrieving PAT analysis: %s", processing_id)
        # Return error status instead of raising exception to provide better UX
        return PATAnalysisResponse(
            processing_id=processing_id,
            status="failed",
            message=f"Error retrieving analysis: {e!s}",
            analysis_date=None,
            pat_features=None,
            activity_embedding=None,
            metadata={},
        )


def _get_analysis_repository() -> DynamoDBHealthDataRepository:
    """Get analysis repository for retrieving stored results."""
    # TODO: Replace with proper dependency injection
    table_name = os.getenv("DYNAMODB_TABLE_NAME", "clarity-health-data")
    region = os.getenv("AWS_REGION", "us-east-1")
    return DynamoDBHealthDataRepository(table_name=table_name, region=region)


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="PAT Service Health Check",
    description="Check the health status of the PAT analysis service",
)
async def health_check(
    inference_engine: AsyncInferenceEngine = Depends(get_pat_inference_engine),
) -> HealthCheckResponse:
    """Comprehensive health check for the PAT analysis service.

    Verifies:
    - Inference engine status
    - PAT model availability
    - System performance metrics
    """
    try:
        # Get inference engine stats
        engine_stats = inference_engine.get_stats()

        # Get PAT model info
        pat_service = inference_engine.pat_service
        model_info = {
            "initialized": pat_service.is_loaded,
            "model_type": "PAT",
            "version": "1.0.0",
        }

        return HealthCheckResponse(
            service="PAT Analysis API",
            status="healthy",
            timestamp=datetime.now(UTC).isoformat(),
            inference_engine=engine_stats,
            pat_model=model_info,
        )

    except Exception as e:
        logger.exception("Health check failed")
        return HealthCheckResponse(
            service="PAT Analysis API",
            status="unhealthy",
            timestamp=datetime.now(UTC).isoformat(),
            inference_engine={"error": str(e)},
            pat_model={"error": "Unable to check model status"},
        )


@router.get(
    "/models/info",
    summary="Get PAT Model Information",
    description="Get information about the available PAT models and current configuration",
)
async def get_model_info(
    _current_user: AuthenticatedUser,
    inference_engine: AsyncInferenceEngine = Depends(get_pat_inference_engine),
) -> dict[str, Any]:
    """Get detailed information about PAT model configuration and capabilities."""
    try:
        # Get PAT service info
        pat_service = inference_engine.pat_service

        # Get inference engine stats
        engine_stats = inference_engine.get_stats()

        model_info = {
            "model_type": "PAT",
            "version": "1.0.0",
            "initialized": pat_service.is_loaded,
            "capabilities": [
                "actigraphy_analysis",
                "sleep_pattern_detection",
                "circadian_rhythm_analysis",
                "activity_classification",
            ],
            "input_requirements": {
                "sampling_rate": "1 sample per minute",
                "duration": "24-168 hours",
                "data_format": "ActigraphyDataPoint",
            },
            "performance": engine_stats,
        }
    except Exception as e:
        logger.exception("Model info request failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to retrieve model information: {e!s}",
        ) from e
    else:
        return model_info
