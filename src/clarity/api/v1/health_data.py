"""CLARITY Digital Twin Platform - Health Data API.

RESTful API endpoints for health data upload, processing, and retrieval.
Implements enterprise-grade security, validation, and HIPAA compliance.
"""

from datetime import UTC, datetime
import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from clarity.auth import Permission, UserContext, get_current_user, require_auth
from clarity.models.health_data import HealthDataResponse, HealthDataUpload
from clarity.services.health_data_service import (
    HealthDataService,
    HealthDataServiceError,
)
from clarity.storage.firestore_client import FirestoreClient

# Configure logger
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/health-data", tags=["Health Data"])


# Dependency injection for services
async def get_health_data_service() -> HealthDataService:
    """Get health data service instance."""
    firestore_client = FirestoreClient(project_id="your-project-id")
    return HealthDataService(firestore_client)


@router.post(
    "/upload",
    summary="Upload Health Data",
    description="Upload health metrics for processing and analysis by the CLARITY digital twin platform.",
)
@require_auth(permissions=[Permission.WRITE_OWN_DATA])
async def upload_health_data(
    health_data: HealthDataUpload,
    current_user: UserContext = Depends(get_current_user),
    service: HealthDataService = Depends(get_health_data_service),
) -> HealthDataResponse:
    """Upload health data for processing."""
    try:
        logger.info("Health data upload requested by user: %s", current_user.user_id)

        # Validate user owns the data
        if str(health_data.user_id) != current_user.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Cannot upload data for another user",
            )

        # Process health data
        response = await service.process_health_data(health_data)

        logger.info("Health data uploaded successfully: %s", response.processing_id)
        return response

    except HealthDataServiceError as e:
        logger.exception("Health data service error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Health data processing failed: {e.message}",
        ) from None
    except Exception as e:
        logger.exception("Unexpected error in health data upload: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from None


@router.get(
    "/processing/{processing_id}",
    summary="Get Processing Status",
    description="Check the processing status of a health data upload using its processing ID.",
)
@require_auth(permissions=[Permission.READ_OWN_DATA])
async def get_processing_status(
    processing_id: UUID,
    current_user: UserContext = Depends(get_current_user),
    service: HealthDataService = Depends(get_health_data_service),
) -> dict[str, Any]:
    """Get processing status for a health data upload."""
    try:
        logger.debug(
            "Processing status requested: %s by user: %s",
            processing_id,
            current_user.user_id,
        )

        status_info = await service.get_processing_status(
            processing_id=str(processing_id), user_id=current_user.user_id
        )

        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Processing job not found"
            )

        return status_info

    except HealthDataServiceError as e:
        logger.exception("Health data service error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from None
    except Exception as e:
        logger.exception("Unexpected error getting processing status: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from None


@router.get("/health-data")
@require_auth(permissions=[Permission.READ_OWN_DATA])
async def get_health_data(
    current_user: UserContext = Depends(get_current_user),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    metric_type: str | None = Query(None, description="Filter by metric type"),
    start_date: datetime | None = Query(None, description="Filter from date"),
    end_date: datetime | None = Query(None, description="Filter to date"),
    service: HealthDataService = Depends(get_health_data_service),
) -> dict[str, Any]:
    """Retrieve user's health data with filtering and pagination."""
    try:
        logger.debug(
            "Health data retrieval requested by user: %s", current_user.user_id
        )

        health_data = await service.get_user_health_data(
            user_id=current_user.user_id,
            limit=limit,
            offset=offset,
            metric_type=metric_type,
            start_date=start_date,
            end_date=end_date,
        )

        logger.info(
            "Retrieved %s health data records for user: %s",
            len(health_data.get("metrics", [])),
            current_user.user_id,
        )
        return health_data

    except HealthDataServiceError as e:
        logger.exception("Health data service error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from None
    except Exception as e:
        logger.exception("Unexpected error retrieving health data: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from None


@router.delete("/health-data/{processing_id}")
@require_auth(permissions=[Permission.WRITE_OWN_DATA])
async def delete_health_data(
    processing_id: UUID,
    current_user: UserContext = Depends(get_current_user),
    service: HealthDataService = Depends(get_health_data_service),
) -> dict[str, str]:
    """Delete user's health data by processing ID."""
    try:
        logger.info(
            "Health data deletion requested: %s by user: %s",
            processing_id,
            current_user.user_id,
        )

        success = await service.delete_health_data(
            user_id=current_user.user_id, processing_id=str(processing_id)
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Processing job not found or access denied",
            )

        logger.info("Health data deleted successfully: %s", processing_id)
        return {"message": "Health data deleted successfully"}

    except HealthDataServiceError as e:
        logger.exception("Health data service error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from None
    except Exception as e:
        logger.exception("Unexpected error deleting health data: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from None


@router.get("/health")
async def health_check(
    service: HealthDataService = Depends(get_health_data_service),
) -> dict[str, Any]:
    """Health check endpoint for the health data service."""
    try:
        # Basic health check
        health_status = {
            "status": "healthy",
            "service": "health-data-api",
            "timestamp": datetime.now(UTC).isoformat(),
        }
        return health_status

    except Exception as e:
        logger.exception("Health check failed: %s", e)
        return {
            "status": "unhealthy",
            "service": "health-data-api",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }
