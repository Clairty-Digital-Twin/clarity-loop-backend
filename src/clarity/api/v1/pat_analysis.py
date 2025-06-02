"""FastAPI endpoints for PAT (Pretrained Actigraphy Transformer) Analysis Service.

This module provides REST API endpoints for actigraphy analysis using the PAT model,
enabling the core "chat with your actigraphy" vertical slice functionality.

Endpoints:
- POST /analyze - Submit actigraphy data for PAT analysis
- GET /analysis/{analysis_id} - Retrieve analysis results
- POST /analyze-step-data - Submit Apple HealthKit step data for analysis  
- GET /health - Service health check
"""

import logging
import uuid
from datetime import datetime, UTC
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from clarity.auth.firebase_auth import get_current_user
from clarity.ml.pat_service import ActigraphyInput, ActigraphyAnalysis
from clarity.ml.inference_engine import AsyncInferenceEngine, get_inference_engine
from clarity.ml.preprocessing import ActigraphyDataPoint
from clarity.ml.proxy_actigraphy import StepCountData, create_proxy_actigraphy_transformer

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/pat", tags=["PAT Analysis"])


class StepDataRequest(BaseModel):
    """Request for Apple HealthKit step data analysis."""
    
    step_counts: List[int] = Field(
        description="Minute-by-minute step counts (10,080 values for 1 week)",
        min_length=1,
        max_length=20160  # 2 weeks max
    )
    timestamps: List[datetime] = Field(
        description="Corresponding timestamps for each step count"
    )
    user_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional user demographics (age_group, sex, etc.)"
    )
    
    @validator('timestamps')
    def validate_timestamps_match_steps(cls, v: List[datetime], values: Dict[str, Any]) -> List[datetime]:
        """Ensure timestamps match step count length."""
        if 'step_counts' in values and len(v) != len(values['step_counts']):
            raise ValueError("Timestamps length must match step_counts length")
        return v


class DirectActigraphyRequest(BaseModel):
    """Request for direct actigraphy data analysis."""
    
    data_points: List[Dict[str, Any]] = Field(
        description="Actigraphy data points with timestamp and value"
    )
    sampling_rate: float = Field(
        default=1.0,
        description="Samples per minute",
        gt=0
    )
    duration_hours: int = Field(
        default=24,
        description="Duration in hours",
        ge=1,
        le=168  # 1 week max
    )


class AnalysisResponse(BaseModel):
    """Response for analysis requests."""
    
    analysis_id: str = Field(description="Unique analysis identifier")
    status: str = Field(description="Analysis status: processing, completed, failed")
    analysis: Optional[ActigraphyAnalysis] = Field(
        default=None,
        description="Analysis results (available when status=completed)"
    )
    processing_time_ms: Optional[float] = Field(
        default=None,
        description="Processing time in milliseconds"
    )
    message: Optional[str] = Field(
        default=None,
        description="Status message or error description"
    )
    cached: bool = Field(
        default=False,
        description="Whether result was retrieved from cache"
    )


class HealthCheckResponse(BaseModel):
    """PAT service health check response."""
    
    service: str
    status: str
    timestamp: str
    version: str = "1.0.0"
    inference_engine: Dict[str, Any]
    pat_model: Dict[str, Any]


async def get_pat_inference_engine() -> AsyncInferenceEngine:
    """Dependency to get the PAT inference engine."""
    return await get_inference_engine()


@router.post(
    "/analyze-step-data",
    response_model=AnalysisResponse,
    summary="Analyze Apple HealthKit Step Data",
    description="Submit Apple HealthKit step count data for PAT analysis using proxy actigraphy transformation"
)
async def analyze_step_data(
    request: StepDataRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
    inference_engine: AsyncInferenceEngine = Depends(get_pat_inference_engine)
) -> AnalysisResponse:
    """
    Analyze Apple HealthKit step data using PAT model with proxy actigraphy transformation.
    
    This endpoint is the core of the "chat with your actigraphy" vertical slice,
    enabling analysis of Apple HealthKit step data through:
    1. Proxy actigraphy transformation
    2. PAT model inference 
    3. Clinical insight generation
    """
    analysis_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Starting step data analysis for user {user_id}, analysis_id {analysis_id}")
        
        # Transform step data to proxy actigraphy
        transformer = create_proxy_actigraphy_transformer()
        
        step_data = StepCountData(
            user_id=user_id,
            upload_id=analysis_id,
            step_counts=[float(count) for count in request.step_counts],  # Convert to float
            timestamps=request.timestamps
        )
        
        # Generate proxy actigraphy vector
        proxy_result = transformer.transform_step_data(step_data)
        logger.info(f"Proxy actigraphy transformation complete: quality={proxy_result.quality_score:.3f}")
        
        # Convert to ActigraphyDataPoint format for PAT model
        actigraphy_points = [
            ActigraphyDataPoint(
                timestamp=request.timestamps[i],
                value=float(proxy_result.vector[i])
            )
            for i in range(len(proxy_result.vector))
        ]
        
        # Create PAT input
        pat_input = ActigraphyInput(
            user_id=user_id,
            data_points=actigraphy_points,
            sampling_rate=1.0,  # 1 sample per minute
            duration_hours=len(proxy_result.vector) // 60
        )
        
        # Submit for async inference
        inference_response = await inference_engine.predict(
            input_data=pat_input,
            request_id=analysis_id,
            timeout_seconds=60.0,
            cache_enabled=True
        )
        
        logger.info(f"PAT analysis complete for {analysis_id}, cached={inference_response.cached}")
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            status="completed",
            analysis=inference_response.analysis,
            processing_time_ms=inference_response.processing_time_ms,
            message="Analysis completed successfully",
            cached=inference_response.cached
        )
        
    except Exception as e:
        logger.error(f"Step data analysis failed for {analysis_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "analysis_id": analysis_id,
                "error": "Analysis failed",
                "message": str(e)
            }
        )


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analyze Direct Actigraphy Data",
    description="Submit preprocessed actigraphy data for PAT analysis"
)
async def analyze_actigraphy_data(
    request: DirectActigraphyRequest,
    user_id: str = Depends(get_current_user),
    inference_engine: AsyncInferenceEngine = Depends(get_pat_inference_engine)
) -> AnalysisResponse:
    """
    Analyze preprocessed actigraphy data using the PAT model.
    
    For direct actigraphy data that doesn't require proxy transformation.
    """
    analysis_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Starting direct actigraphy analysis for user {user_id}, analysis_id {analysis_id}")
        
        # Convert request data to ActigraphyDataPoint format
        actigraphy_points = [
            ActigraphyDataPoint(
                timestamp=datetime.fromisoformat(dp["timestamp"]),
                value=float(dp["value"])
            )
            for dp in request.data_points
        ]
        
        # Create PAT input
        pat_input = ActigraphyInput(
            user_id=user_id,
            data_points=actigraphy_points,
            sampling_rate=request.sampling_rate,
            duration_hours=request.duration_hours
        )
        
        # Submit for async inference
        inference_response = await inference_engine.predict(
            input_data=pat_input,
            request_id=analysis_id,
            timeout_seconds=60.0,
            cache_enabled=True
        )
        
        logger.info(f"Direct actigraphy analysis complete for {analysis_id}, cached={inference_response.cached}")
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            status="completed",
            analysis=inference_response.analysis,
            processing_time_ms=inference_response.processing_time_ms,
            message="Analysis completed successfully",
            cached=inference_response.cached
        )
        
    except Exception as e:
        logger.error(f"Direct actigraphy analysis failed for {analysis_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "analysis_id": analysis_id,
                "error": "Analysis failed", 
                "message": str(e)
            }
        )


@router.get(
    "/analysis/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Get Analysis Results",
    description="Retrieve analysis results by analysis ID"
)
async def get_analysis_results(
    analysis_id: str,
    user_id: str = Depends(get_current_user)
) -> AnalysisResponse:
    """
    Retrieve analysis results by analysis ID.
    
    Note: This is a simplified implementation. In production, you would
    store analysis results in a database and retrieve them here.
    """
    # For now, return a message indicating this is not implemented
    # In production, implement database lookup
    return AnalysisResponse(
        analysis_id=analysis_id,
        status="not_found",
        message="Analysis result lookup not implemented. Results are returned synchronously from analysis endpoints."
    )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="PAT Service Health Check",
    description="Check the health status of the PAT analysis service"
)
async def health_check(
    inference_engine: AsyncInferenceEngine = Depends(get_pat_inference_engine)
) -> HealthCheckResponse:
    """
    Comprehensive health check for the PAT analysis service.
    
    Returns status of all components: inference engine, PAT model, caching, etc.
    """
    try:
        # Get inference engine health
        engine_health = await inference_engine.health_check()
        
        # Get PAT service health (from the engine's PAT service)
        pat_health = engine_health.get("pat_service", {})
        
        # Determine overall status
        overall_status = "healthy"
        if engine_health.get("status") != "healthy":
            overall_status = "degraded"
        if not pat_health.get("model_loaded", False):
            overall_status = "unhealthy"
            
        return HealthCheckResponse(
            service="PAT Analysis API",
            status=overall_status,
            timestamp=datetime.now(UTC).isoformat(),
            inference_engine=engine_health,
            pat_model=pat_health
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheckResponse(
            service="PAT Analysis API", 
            status="unhealthy",
            timestamp=datetime.now(UTC).isoformat(),
            inference_engine={"error": str(e)},
            pat_model={"error": "Unable to check PAT model status"}
        )


@router.get(
    "/models/info", 
    summary="Get PAT Model Information",
    description="Get information about the available PAT models and current configuration"
)
async def get_model_info(
    user_id: str = Depends(get_current_user),
    inference_engine: AsyncInferenceEngine = Depends(get_pat_inference_engine)
) -> Dict[str, Any]:
    """Get detailed information about PAT model configuration and capabilities."""
    try:
        health_info = await inference_engine.health_check()
        pat_info = health_info.get("pat_service", {})
        
        return {
            "model_name": "Dartmouth PAT (Pretrained Actigraphy Transformer)",
            "version": "v1.0",
            "model_size": pat_info.get("model_size", "medium"),
            "device": pat_info.get("device", "cpu"),
            "capabilities": [
                "Sleep stage classification (wake, light, deep, REM)",
                "Sleep efficiency calculation",
                "Circadian rhythm analysis", 
                "Depression risk assessment",
                "Activity fragmentation analysis"
            ],
            "input_requirements": {
                "sampling_rate": "1 sample per minute",
                "duration": "24 hours (1440 samples) recommended",
                "data_format": "Activity counts or proxy actigraphy from step data"
            },
            "proxy_actigraphy": {
                "supported": True,
                "source": "Apple HealthKit step counts",
                "transformation": "Square root + NHANES z-score normalization"
            }
        }
        
    except Exception as e:
        logger.error(f"Model info request failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to retrieve model information: {str(e)}"
        ) 