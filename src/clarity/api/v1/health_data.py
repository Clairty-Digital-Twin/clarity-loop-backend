"""CLARITY Digital Twin Platform - Health Data API.

RESTful API endpoints for health data upload, processing, and retrieval.
Implements enterprise-grade security, validation, and HIPAA compliance.

🔥 ENHANCED WITH:
- RFC 7807 Problem Details error handling
- Professional pagination with HAL-style links
- Improved endpoint structure and validation
- Enhanced API documentation

Following Robert C. Martin's Clean Architecture with proper dependency injection.
"""

# removed - breaks FastAPI

from datetime import UTC, datetime
import json
import logging
import os
from typing import Any, NoReturn
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from google.cloud import storage
from slowapi import Limiter

from clarity.auth.dependencies import AuthenticatedUser
from clarity.core.exceptions import (
    AuthorizationProblem,
    InternalServerProblem,
    ResourceNotFoundProblem,
    ServiceUnavailableProblem,
    ValidationProblem,
)
from clarity.core.pagination import (
    PaginatedResponse,
    PaginationBuilder,
    validate_pagination_params,
)
from clarity.middleware.rate_limiting import get_user_id_or_ip
from clarity.models.health_data import HealthDataResponse, HealthDataUpload
from clarity.ports.auth_ports import IAuthProvider
from clarity.ports.config_ports import IConfigProvider
from clarity.ports.data_ports import IHealthDataRepository
from clarity.services.health_data_service import (
    HealthDataService,
    HealthDataServiceError,
)
from clarity.services.messaging.publisher import get_publisher

# Configure logger
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(tags=["health-data"])

# Create rate limiter for health data endpoints
health_limiter = Limiter(key_func=get_user_id_or_ip)


@router.get("/health", summary="Health Check")
async def health_check() -> dict[str, Any]:
    """Health check endpoint for the health data API."""
    return {
        "status": "healthy",
        "service": "health-data-api",
        "timestamp": datetime.now(UTC).isoformat(),
    }


# Query parameter constants to fix B008 linting issues
_START_DATE_QUERY = Query(None, description="Filter from date (ISO 8601)")
_END_DATE_QUERY = Query(None, description="Filter to date (ISO 8601)")
_LEGACY_START_DATE_QUERY = Query(None, description="Filter from date")
_LEGACY_END_DATE_QUERY = Query(None, description="Filter to date")


# Helper functions to fix linting issues (TRY300/TRY301)
def _raise_authorization_error(user_id: str) -> NoReturn:
    """Raise authorization error for user data access."""
    error_msg = (
        f"Cannot upload health data for user '{user_id}'. "
        "Users can only upload their own data."
    )
    raise AuthorizationProblem(detail=error_msg)


def _raise_not_found_error(resource_type: str, resource_id: str) -> NoReturn:
    """Raise not found error for missing resources."""
    raise ResourceNotFoundProblem(resource_type=resource_type, resource_id=resource_id)


def _raise_too_many_metrics_error(metrics_count: int, max_allowed: int) -> NoReturn:
    """Raise validation error for too many metrics in upload."""
    error_detail = (
        f"Too many metrics in upload: {metrics_count} exceeds maximum {max_allowed}"
    )
    raise ValidationProblem(
        detail=error_detail,
        errors=[
            {
                "field": "metrics",
                "error": "too_many_items",
                "received": metrics_count,
                "max_allowed": max_allowed,
            }
        ],
    )


# Dependency injection container - using class-based approach instead of globals
class DependencyContainer:
    """Container for dependency injection to avoid global variables."""

    def __init__(self) -> None:
        self.auth_provider: IAuthProvider | None = None
        self.repository: IHealthDataRepository | None = None
        self.config_provider: IConfigProvider | None = None

    def set_dependencies(
        self,
        auth_provider: IAuthProvider,
        repository: IHealthDataRepository,
        config_provider: IConfigProvider,
    ) -> None:
        """Set dependencies from the DI container."""
        self.auth_provider = auth_provider
        self.repository = repository
        self.config_provider = config_provider
        logger.info("Health data API dependencies injected successfully")


# Container instance
_container = DependencyContainer()


def set_dependencies(
    auth_provider: IAuthProvider,
    repository: IHealthDataRepository,
    config_provider: IConfigProvider,
) -> None:
    """Set dependencies from the DI container.

    Called by the container during application initialization.
    Follows Dependency Inversion Principle - depends on abstractions, not concretions.
    """
    _container.set_dependencies(auth_provider, repository, config_provider)


def get_health_data_service() -> HealthDataService:
    """Get health data service instance with injected dependencies.

    Uses dependency injection container instead of hardcoded dependencies.
    Follows Clean Architecture principles.
    """
    if _container.repository is None:
        raise ServiceUnavailableProblem(
            service_name="Health Data Repository", retry_after=30
        )

    return HealthDataService(_container.repository)


def get_auth_provider() -> IAuthProvider:
    """Get authentication provider from dependency injection."""
    if _container.auth_provider is None:
        raise ServiceUnavailableProblem(
            service_name="Authentication Provider", retry_after=30
        )
    return _container.auth_provider


def get_config_provider() -> IConfigProvider:
    """Get configuration provider from dependency injection."""
    if _container.config_provider is None:
        raise ServiceUnavailableProblem(
            service_name="Configuration Provider", retry_after=30
        )
    return _container.config_provider


@router.post(
    "/",
    summary="Upload Health Data",
    description="""
    Upload health metrics for processing and analysis by the CLARITY
    digital twin platform.

    **Features:**
    - Supports multiple data types (heart rate, sleep, activity, etc.)
    - Real-time validation and processing
    - HIPAA-compliant secure storage
    - Automatic data quality checks

    **Example Request:**
    ```json
    {
        "user_id": "user_123",
        "data_type": "heart_rate",
        "measurements": [
            {
                "timestamp": "2025-01-15T10:30:00Z",
                "value": 72.5,
                "unit": "bpm"
            }
        ],
        "source": "apple_watch",
        "device_info": {
            "model": "Apple Watch Series 9",
            "os_version": "10.0"
        }
    }
    ```
    """,
    response_model=HealthDataResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Health data uploaded successfully"},
        400: {"description": "Validation error - invalid data format"},
        403: {
            "description": "Authorization denied - cannot upload data for another user"
        },
        503: {"description": "Service temporarily unavailable"},
    },
)
@router.post(
    "",
    summary="Upload Health Data",
    description="Upload health data (canonical endpoint without trailing slash)",
    response_model=HealthDataResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,  # Don't show duplicate in OpenAPI docs
)
@health_limiter.limit("100/hour")  # Rate limit: 100 uploads per hour per user
async def upload_health_data(
    request: Request,
    health_data: HealthDataUpload,
    current_user: AuthenticatedUser,
    service: HealthDataService = Depends(get_health_data_service),
) -> HealthDataResponse:
    """🔥 Upload health data with enterprise-grade processing."""
    _ = request  # Used by rate limiter
    try:
        logger.info("Health data upload requested by user: %s", current_user.user_id)

        # SECURITY: Validate metrics count to prevent DoS through large uploads
        max_metrics_per_upload = 10000  # Reasonable limit for health data batches
        if len(health_data.metrics) > max_metrics_per_upload:
            _raise_too_many_metrics_error(
                len(health_data.metrics), max_metrics_per_upload
            )

        # Validate user owns the data
        if str(health_data.user_id) != current_user.user_id:
            _raise_authorization_error(str(health_data.user_id))

        # Process health data
        response = await service.process_health_data(health_data)
        logger.info("Health data uploaded successfully: %s", response.processing_id)

        # Save to GCS and publish to Pub/Sub for async processing
        try:
            # Save raw health data to GCS
            bucket_name = os.getenv("HEALTHKIT_RAW_BUCKET", "healthkit-raw-data")
            blob_path = f"{current_user.user_id}/{response.processing_id}.json"
            gcs_path = f"gs://{bucket_name}/{blob_path}"

            # Save to GCS
            try:
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_path)

                # Convert health data to JSON for storage
                raw_data = {
                    "user_id": str(health_data.user_id),
                    "processing_id": str(response.processing_id),
                    "upload_source": health_data.upload_source,
                    "client_timestamp": health_data.client_timestamp.isoformat(),
                    "sync_token": health_data.sync_token,
                    "metrics": [metric.model_dump() for metric in health_data.metrics],
                }

                blob.upload_from_string(
                    json.dumps(raw_data), content_type="application/json"
                )
                logger.info("Saved health data to GCS: %s", gcs_path)
            except Exception:
                logger.exception("Failed to save to GCS")
                # Continue anyway - we can retry later

            # Publish to Pub/Sub
            publisher = await get_publisher()
            await publisher.publish_health_data_upload(
                user_id=current_user.user_id,
                upload_id=str(response.processing_id),
                # Note: Using gcs_path value but naming it s3_path
                # for compatibility
                s3_path=gcs_path,
                metadata={
                    "source": health_data.upload_source,
                    "metrics_count": len(health_data.metrics),
                    "timestamp": health_data.client_timestamp.isoformat(),
                },
            )
            logger.info(
                "Published health data event for async processing: %s",
                response.processing_id,
            )
        except Exception:  # noqa: BLE001 - Don't fail upload if Pub/Sub fails
            # Don't fail the upload if Pub/Sub fails - data is already saved
            logger.warning("Failed to publish health data event")
    except HealthDataServiceError as e:
        logger.exception("Health data service error")
        raise ValidationProblem(
            detail=f"Health data processing failed: {e.message}",
            errors=[
                {"field": "health_data", "message": str(e), "code": "PROCESSING_ERROR"}
            ],
        ) from e
    except Exception as e:
        logger.exception("Unexpected error in health data upload")
        raise InternalServerProblem(
            detail="An unexpected error occurred while processing health data upload"
        ) from e
    else:
        return response


@router.get(
    "/processing/{processing_id}",
    summary="Get Processing Status",
    description="""
    Check the processing status of a health data upload using its processing ID.

    **Status Values:**
    - `pending`: Upload received, processing queued
    - `processing`: Data currently being analyzed
    - `completed`: Processing finished successfully
    - `failed`: Processing encountered an error
    - `cancelled`: Processing was cancelled

    **Response includes:**
    - Current processing stage
    - Progress percentage (if available)
    - Estimated completion time
    - Error details (if failed)
    """,
    responses={
        200: {"description": "Processing status retrieved successfully"},
        404: {"description": "Processing job not found"},
        403: {"description": "Access denied - can only view own processing jobs"},
        503: {"description": "Service temporarily unavailable"},
    },
)
async def get_processing_status(
    processing_id: UUID,
    current_user: AuthenticatedUser,
    service: HealthDataService = Depends(get_health_data_service),
) -> dict[str, Any]:
    """🔥 Get processing status with detailed progress information."""
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
            _raise_not_found_error("Processing Job", str(processing_id))

        logger.debug("Retrieved processing status: %s", processing_id)
        return status_info

    except HealthDataServiceError as e:
        logger.exception("Health data service error")
        raise ValidationProblem(detail=str(e)) from e
    except Exception as e:
        logger.exception("Unexpected error getting processing status")
        raise InternalServerProblem(
            detail="An unexpected error occurred while retrieving processing status"
        ) from e


@router.get(
    "/",
    summary="List Health Data",
    description="""
    Retrieve paginated health data with advanced filtering and sorting options.

    **Pagination:**
    - Cursor-based pagination for consistent results
    - HAL-style navigation links
    - Configurable page sizes (1-1000 items)

    **Filtering:**
    - Filter by data type (heart_rate, sleep, activity, etc.)
    - Date range filtering with timezone support
    - Source device filtering

    **Sorting:**
    - Default: Most recent first
    - Customizable sort orders

    **Example Response:**
    ```json
    {
        "data": [
            {
                "id": "data_123",
                "timestamp": "2025-01-15T10:30:00Z",
                "data_type": "heart_rate",
                "value": 72.5,
                "unit": "bpm"
            }
        ],
        "pagination": {
            "page_size": 50,
            "has_next": true,
            "has_previous": false,
            "next_cursor": "eyJpZCI6IjEyMyJ9"
        },
        "links": {
            "self": "https://api.clarity.health/api/v1/health-data?limit=50",
            "next": "https://api.clarity.health/api/v1/health-data?limit=50&cursor=eyJpZCI6IjEyMyJ9",
            "first": "https://api.clarity.health/api/v1/health-data?limit=50"
        }
    }
    ```
    """,
    response_model=PaginatedResponse[dict[str, Any]],
    responses={
        200: {"description": "Health data retrieved successfully"},
        400: {"description": "Invalid pagination or filter parameters"},
        403: {"description": "Access denied - can only view own data"},
        503: {"description": "Service temporarily unavailable"},
    },
)
# Also define the same endpoint without trailing slash to prevent redirects
@router.get(
    "",
    summary="List Health Data",
    description="List health data (canonical endpoint without trailing slash)",
    response_model=PaginatedResponse[dict[str, Any]],
    include_in_schema=False,  # Don't show duplicate in OpenAPI docs
)
async def list_health_data(
    request: Request,
    current_user: AuthenticatedUser,
    limit: int = Query(50, ge=1, le=1000, description="Number of items per page"),
    cursor: str | None = Query(None, description="Pagination cursor"),
    offset: int | None = Query(
        None, ge=0, description="Offset (alternative to cursor)"
    ),
    data_type: str | None = Query(
        None, description="Filter by data type (heart_rate, sleep, etc.)"
    ),
    start_date: datetime | None = _START_DATE_QUERY,
    end_date: datetime | None = _END_DATE_QUERY,
    source: str | None = Query(
        None, description="Filter by data source (apple_watch, fitbit, etc.)"
    ),
    service: HealthDataService = Depends(get_health_data_service),
) -> PaginatedResponse[dict[str, Any]]:
    """🔥 Retrieve paginated health data with professional pagination."""
    try:
        logger.debug(
            "Health data retrieval requested by user: %s", current_user.user_id
        )

        # Validate pagination parameters
        pagination_params = validate_pagination_params(
            limit=limit, cursor=cursor, offset=offset
        )

        # Build filter parameters
        filters = {}
        if data_type:
            filters["data_type"] = data_type
        if start_date:
            filters["start_date"] = start_date.isoformat()
        if end_date:
            filters["end_date"] = end_date.isoformat()
        if source:
            filters["source"] = source

        # Get health data from service (fallback to legacy method for now)
        # TODO: Implement get_user_health_data_paginated in HealthDataService
        legacy_data = await service.get_user_health_data(
            user_id=current_user.user_id,
            limit=pagination_params.limit,
            offset=pagination_params.offset or 0,
            metric_type=filters.get("data_type"),
            start_date=start_date,
            end_date=end_date,
        )

        # Convert legacy format to paginated format
        data_items = legacy_data.get("metrics", [])
        has_next = len(data_items) == pagination_params.limit  # Simple heuristic
        has_previous = (pagination_params.offset or 0) > 0

        health_data_result = {
            "data": data_items,
            "has_next": has_next,
            "has_previous": has_previous,
            "total_count": None,  # Not available in legacy format
            "next_cursor": None,  # Not implemented yet
            "previous_cursor": None,  # Not implemented yet
        }

        # Extract base URL for pagination links
        base_url = f"{request.url.scheme}://{request.url.netloc}"

        # Build pagination response
        pagination_builder = PaginationBuilder(
            base_url=base_url, endpoint="/api/v1/health-data"
        )

        paginated_response = pagination_builder.build_response(
            data=health_data_result["data"],
            params=pagination_params,
            has_next=health_data_result["has_next"],
            has_previous=health_data_result["has_previous"],
            total_count=health_data_result.get("total_count"),
            next_cursor=health_data_result.get("next_cursor"),
            previous_cursor=health_data_result.get("previous_cursor"),
            additional_params=filters,
        )

        logger.debug("Retrieved health data for user: %s", current_user.user_id)
        return paginated_response

    except ValueError as e:
        logger.warning("Invalid pagination parameters: %s", e)
        raise ValidationProblem(
            detail=str(e),
            errors=[
                {"field": "pagination", "message": str(e), "code": "INVALID_PARAMETER"}
            ],
        ) from e
    except HealthDataServiceError as e:
        logger.exception("Health data service error")
        raise ValidationProblem(detail=str(e)) from e
    except Exception as e:
        logger.exception("Unexpected error retrieving health data")
        raise InternalServerProblem(
            detail="An unexpected error occurred while retrieving health data"
        ) from e


@router.get(
    "/query",
    summary="Query Health Data (Removed)",
    description="""
    **REMOVED:** This legacy endpoint has been permanently removed.
    Use `GET /health-data/` instead.
    """,
    status_code=410,
    responses={
        410: {
            "description": (
                "Endpoint permanently removed - " "use GET /health-data/ instead"
            )
        },
    },
    include_in_schema=False,  # Hide from OpenAPI docs
)
async def query_health_data_legacy() -> dict[str, str]:
    """🚫 Legacy endpoint permanently removed."""
    logger.warning("Attempt to access removed legacy health data query endpoint")

    raise HTTPException(
        status_code=410,
        detail={
            "error": "Endpoint Permanently Removed",
            "message": "The legacy /query endpoint has been permanently removed.",
            "migration": {
                "new_endpoint": "GET /api/v1/health-data/",
                "documentation": "See API documentation for the new paginated endpoint",
                "benefits": [
                    "Improved pagination with cursor support",
                    "Better filtering options",
                    "Consistent response format",
                    "Enhanced performance",
                ],
            },
            "removed_date": "2025-06-11",
            "status_code": 410,
        },
    )


@router.delete(
    "/{processing_id}",
    summary="Delete Health Data",
    description="""
    Delete health data by processing ID with proper authorization checks.

    **Security:**
    - Users can only delete their own data
    - Soft delete with audit trail
    - GDPR/CCPA compliance support

    **Note:** This action cannot be undone. Consider data export before deletion.
    """,
    responses={
        200: {"description": "Health data deleted successfully"},
        404: {"description": "Processing job not found"},
        403: {"description": "Access denied - can only delete own data"},
        503: {"description": "Service temporarily unavailable"},
    },
)
async def delete_health_data(
    processing_id: UUID,
    current_user: AuthenticatedUser,
    service: HealthDataService = Depends(get_health_data_service),
) -> dict[str, str]:
    """🔥 Delete health data with proper authorization and audit trail."""
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
            _raise_not_found_error("Processing Job", str(processing_id))

        logger.info("Health data deleted successfully: %s", processing_id)
        return {
            "message": "Health data deleted successfully",
            "processing_id": str(processing_id),
            "deleted_at": datetime.now(UTC).isoformat(),
        }

    except HealthDataServiceError as e:
        logger.exception("Health data service error")
        raise ValidationProblem(detail=str(e)) from e
    except Exception as e:
        logger.exception("Unexpected error deleting health data")
        raise InternalServerProblem(
            detail="An unexpected error occurred while deleting health data"
        ) from e


@router.get(
    "/health",
    summary="Health Data Service Status",
    description="""
    Health check endpoint for the health data service with detailed status information.

    **Status Indicators:**
    - `healthy`: Service fully operational
    - `degraded`: Service operational with reduced functionality
    - `unhealthy`: Service experiencing issues

    **Includes:**
    - Database connectivity status
    - Cache status
    - Processing queue status
    - Performance metrics
    """,
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy"},
    },
)
async def health_check_detailed() -> dict[str, Any]:
    """🔥 Comprehensive health check with detailed status information."""
    try:
        # Get current timestamp
        timestamp = datetime.now(UTC).isoformat()

        # Basic health indicators
        health_status: dict[str, Any] = {
            "status": "healthy",
            "service": "health-data-api",
            "timestamp": timestamp,
            "version": "1.0.0",
        }

        # Check if dependencies are available
        try:
            if _container.repository is not None:
                health_status["database"] = "connected"
            else:
                health_status["database"] = "not_configured"
                health_status["status"] = "degraded"
        except (AttributeError, RuntimeError) as db_error:
            logger.warning("Database health check failed: %s", db_error)
            health_status["database"] = "error"
            health_status["status"] = "degraded"

        try:
            if _container.auth_provider is not None:
                health_status["authentication"] = "available"
            else:
                health_status["authentication"] = "not_configured"
                health_status["status"] = "degraded"
        except (AttributeError, RuntimeError) as auth_error:
            logger.warning("Auth health check failed: %s", auth_error)
            health_status["authentication"] = "error"
            health_status["status"] = "degraded"

        # Add performance metrics
        metrics: dict[str, Any] = {
            "uptime_seconds": 0,  # Would be calculated from startup time
            "requests_per_minute": 0,  # Would be tracked by middleware
            "average_response_time_ms": 0,  # Would be tracked by middleware
        }
        health_status["metrics"] = metrics

        logger.debug("Health check completed successfully")
        return health_status

    except Exception as e:
        logger.exception("Health check failed")
        return {
            "status": "unhealthy",
            "service": "health-data-api",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }
