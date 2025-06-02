"""FastAPI endpoints for PAT (Pretrained Actigraphy Transformer) Analysis Service.

This module provides REST API endpoints for actigraphy analysis using the PAT model,
enabling the core "chat with your actigraphy" vertical slice functionality.

Endpoints:
- POST /analyze - Submit actigraphy data for PAT analysis
- GET /analysis/{analysis_id} - Retrieve analysis results
- POST /analyze-step-data - Submit Apple HealthKit step data for analysis
- GET /health - Service health check
"""

from datetime import UTC, datetime
import logging
from typing import Any
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field, validator

from clarity.auth.firebase_auth import get_current_user
from clarity.ml.inference_engine import AsyncInferenceEngine, get_inference_engine
from clarity.ml.pat_service import ActigraphyAnalysis, ActigraphyInput
from clarity.ml.preprocessing import ActigraphyDataPoint
from clarity.ml.proxy_actigraphy import (
    StepCountData,
    create_proxy_actigraphy_transformer,
)

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/pat", tags=["PAT Analysis"])


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
        description="Optional user demographics (age_group, sex, etc.)",
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
    "/analyze-step-data",
    response_model=AnalysisResponse,
    summary="Analyze Apple HealthKit Step Data",
    description="Submit Apple HealthKit step count data for PAT analysis using proxy actigraphy transformation",
)
async def analyze_step_data(
    request: StepDataRequest,
    background_tasks: BackgroundTasks,  # noqa: ARG001
    user_id: str = Depends(get_current_user),
    inference_engine: AsyncInferenceEngine = Depends(  # noqa: B008
        get_pat_inference_engine
    ),
) -> AnalysisResponse:
    """Analyze Apple HealthKit step data using PAT model with proxy actigraphy transformation.

    This endpoint is the core of the "chat with your actigraphy" vertical slice,
    enabling analysis of Apple HealthKit step data through:
    1. Proxy actigraphy transformation
    2. PAT model inference
    3. Clinical insight generation
    """
    analysis_id = str(uuid.uuid4())

    try:
        logger.info(
            "Starting step data analysis for user %s, analysis_id %s",
            user_id,
            analysis_id,
        )

        # Transform step data to proxy actigraphy
        transformer = create_proxy_actigraphy_transformer()

        step_data = StepCountData(
            user_id=user_id,
            upload_id=analysis_id,
            step_counts=[
                float(count) for count in request.step_counts
            ],  # Convert to float
            timestamps=request.timestamps,
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
                timestamp=request.timestamps[i], value=float(proxy_result.vector[i])
            )
            for i in range(len(proxy_result.vector))
        ]

        # Create PAT input
        pat_input = ActigraphyInput(
            user_id=user_id,
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
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analyze Direct Actigraphy Data",
    description="Submit preprocessed actigraphy data for PAT analysis",
)
async def analyze_actigraphy_data(
    request: DirectActigraphyRequest,
    user_id: str = Depends(get_current_user),
    inference_engine: AsyncInferenceEngine = Depends(  # noqa: B008
        get_pat_inference_engine
    ),
) -> AnalysisResponse:
    """Analyze preprocessed actigraphy data using the PAT model.

    This endpoint accepts direct actigraphy data that has already been preprocessed
    and formatted for PAT model analysis.
    """
    analysis_id = str(uuid.uuid4())

    try:
        logger.info(
            "Starting direct actigraphy analysis for user %s, analysis_id %s",
            user_id,
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
            user_id=user_id,
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
    "/analysis/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Get Analysis Results",
    description="Retrieve analysis results by analysis ID",
)
async def get_analysis_results(
    analysis_id: str,
    user_id: str = Depends(get_current_user),  # noqa: ARG001
) -> AnalysisResponse:
    """Retrieve analysis results by analysis ID.

    Note: This is a placeholder implementation. In production, this would
    retrieve results from a database or cache.
    """
    # TODO: Implement actual result retrieval from storage
    # For now, return a placeholder response
    return AnalysisResponse(
        analysis_id=analysis_id,
        status="not_found",
        message="Result retrieval not yet implemented",
    )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="PAT Service Health Check",
    description="Check the health status of the PAT analysis service",
)
async def health_check(
    inference_engine: AsyncInferenceEngine = Depends(  # noqa: B008
        get_pat_inference_engine
    ),
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
    user_id: str = Depends(get_current_user),  # noqa: ARG001
    inference_engine: AsyncInferenceEngine = Depends(  # noqa: B008
        get_pat_inference_engine
    ),
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
