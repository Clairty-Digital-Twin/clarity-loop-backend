"""CLARITY Digital Twin Platform - Health Data API

RESTful endpoints for health data upload, processing, and retrieval.
Implements the Phase 1 vertical slice for health data upload and storage.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from ...auth import Permission, UserContext, get_current_user, require_permission
from ...models.health_data import (
    HealthDataResponse,
    HealthDataUpload,
    HealthMetricType,
    ProcessingStatus,
)
from ...services.health_data_service import HealthDataService, HealthDataServiceError
from ...storage.firestore_client import FirestoreClient

# Configure logger
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(
    prefix="/health-data",
    tags=["Health Data"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"}
    }
)


# Dependency to get health data service
async def get_health_data_service() -> HealthDataService:
    """Dependency to get configured health data service."""
    # TODO: Initialize with proper configuration
    firestore_client = FirestoreClient(
        project_id="clarity-digital-twin",  # Should come from settings
        enable_caching=True
    )

    return HealthDataService(firestore_client=firestore_client)


@router.post(
    "/upload",
    response_model=HealthDataResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload Health Data",
    description="Upload health metrics for processing and analysis by the CLARITY digital twin platform."
)
@require_permission(Permission.WRITE_OWN_DATA)
async def upload_health_data(
    health_data: HealthDataUpload,
    current_user: UserContext = Depends(get_current_user),
    service: HealthDataService = Depends(get_health_data_service)
) -> HealthDataResponse:
    """Upload health data for processing and analysis.
    
    This endpoint accepts health metrics from various sources (HealthKit, wearables, manual entry)
    and initiates asynchronous processing for AI-powered insights generation.
    
    **Security**: Requires authentication and WRITE_OWN_DATA permission.
    **Rate Limiting**: Applied per user to prevent abuse.
    **Data Validation**: Comprehensive clinical-grade validation is performed.
    
    Returns a processing ID that can be used to track the status of the upload.
    """
    try:
        logger.info(f"Health data upload requested by user: {current_user.user_id}")

        # Validate user owns the data
        if str(health_data.user_id) != current_user.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot upload data for another user"
            )

        # Process the health data upload
        response = await service.upload_health_data(
            health_data=health_data,
            user_id=current_user.user_id
        )

        logger.info(f"Health data uploaded successfully: {response.processing_id}")
        return response

    except HealthDataServiceError as e:
        logger.error(f"Health data service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in health data upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during health data upload"
        )


@router.get(
    "/{processing_id}/status",
    summary="Get Processing Status",
    description="Check the processing status of a health data upload using its processing ID."
)
@require_permission(Permission.READ_OWN_DATA)
async def get_processing_status(
    processing_id: UUID,
    current_user: UserContext = Depends(get_current_user),
    service: HealthDataService = Depends(get_health_data_service)
) -> dict[str, Any]:
    """Get the processing status of a health data upload.
    
    **Security**: Users can only access status for their own uploads.
    **Real-time Updates**: Status is updated in real-time as processing progresses.
    
    Returns detailed information about the processing pipeline progress.
    """
    try:
        logger.debug(f"Processing status requested: {processing_id} by user: {current_user.user_id}")

        status_info = await service.get_processing_status(
            processing_id=str(processing_id),
            user_id=current_user.user_id
        )

        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Processing ID not found or access denied"
            )

        return status_info

    except HealthDataServiceError as e:
        logger.error(f"Health data service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error getting processing status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error retrieving processing status"
        )


@router.get(
    "/",
    summary="Get User Health Data",
    description="Retrieve health data for the authenticated user with optional filtering."
)
@require_permission(Permission.READ_OWN_DATA)
async def get_user_health_data(
    current_user: UserContext = Depends(get_current_user),
    service: HealthDataService = Depends(get_health_data_service),
    metric_type: HealthMetricType | None = None,
    limit: int = 100
) -> list[dict[str, Any]]:
    """Retrieve health data for the authenticated user.
    
    **Security**: Users can only access their own health data.
    **Filtering**: Optional filtering by metric type and date range.
    **Pagination**: Results are paginated to prevent large responses.
    
    Returns a list of health data records matching the criteria.
    """
    try:
        logger.debug(f"Health data retrieval requested by user: {current_user.user_id}")

        health_data = await service.get_user_health_data(
            user_id=current_user.user_id,
            metric_type=metric_type,
            limit=min(limit, 1000)  # Cap at 1000 records
        )

        logger.info(f"Retrieved {len(health_data)} health data records for user: {current_user.user_id}")
        return health_data

    except HealthDataServiceError as e:
        logger.error(f"Health data service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error retrieving health data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error retrieving health data"
        )


@router.delete(
    "/{processing_id}",
    summary="Delete Health Data",
    description="Delete specific health data upload (GDPR compliance)."
)
@require_permission(Permission.WRITE_OWN_DATA)
async def delete_health_data(
    processing_id: UUID,
    current_user: UserContext = Depends(get_current_user),
    service: HealthDataService = Depends(get_health_data_service)
) -> dict[str, str]:
    """Delete specific health data upload.
    
    **Security**: Users can only delete their own data.
    **GDPR Compliance**: Supports right to erasure requirements.
    **Audit Trail**: Deletion is logged for compliance purposes.
    
    Returns confirmation of successful deletion.
    """
    try:
        logger.info(f"Health data deletion requested: {processing_id} by user: {current_user.user_id}")

        success = await service.delete_user_health_data(
            user_id=current_user.user_id,
            processing_id=str(processing_id)
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Health data not found or access denied"
            )

        logger.info(f"Health data deleted successfully: {processing_id}")
        return {"message": "Health data deleted successfully"}

    except HealthDataServiceError as e:
        logger.error(f"Health data service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error deleting health data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error deleting health data"
        )


@router.get(
    "/health",
    summary="Health Check",
    description="Health check endpoint for the health data service."
)
async def health_check(
    service: HealthDataService = Depends(get_health_data_service)
) -> dict[str, Any]:
    """Health check for the health data service.
    
    **Public Endpoint**: No authentication required.
    **Monitoring**: Used by load balancers and monitoring systems.
    
    Returns service health status and statistics.
    """
    try:
        health_status = await service.health_check()
        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2025-06-01T18:37:00Z"  # This should be dynamic
        }
