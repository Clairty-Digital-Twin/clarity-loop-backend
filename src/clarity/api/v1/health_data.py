"""CLARITY Digital Twin Platform - Health Data API.

RESTful API endpoints for health data upload, processing, and retrieval.
Implements enterprise-grade security, validation, and HIPAA compliance.

Following Robert C. Martin's Clean Architecture with proper dependency injection.
"""

from datetime import UTC, datetime
import logging
from typing import Any, NoReturn
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from clarity.auth import Permission, UserContext, get_current_user, require_auth
from clarity.core.interfaces import (
    IAuthProvider,
    IConfigProvider,
    IHealthDataRepository,
)
from clarity.models.health_data import HealthDataResponse, HealthDataUpload
from clarity.services.health_data_service import (
    HealthDataService,
    HealthDataServiceError,
)

# Configure logger
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/health-data", tags=["Health Data"])


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
        msg = "Health data repository not injected. Container not initialized?"
        raise RuntimeError(msg)

    return HealthDataService(_container.repository)


def get_auth_provider() -> IAuthProvider:
    """Get authentication provider from dependency injection."""
    if _container.auth_provider is None:
        msg = "Auth provider not injected. Container not initialized?"
        raise RuntimeError(msg)
    return _container.auth_provider


def get_config_provider() -> IConfigProvider:
    """Get configuration provider from dependency injection."""
    if _container.config_provider is None:
        msg = "Config provider not injected. Container not initialized?"
        raise RuntimeError(msg)
    return _container.config_provider


# Exception helper functions to satisfy TRY301 linting rule
def _raise_forbidden_access() -> NoReturn:
    """Raise forbidden access exception."""
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Access denied: Cannot upload data for another user",
    )


def _raise_service_error(message: str) -> NoReturn:
    """Raise service error exception."""
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=message,
    )


def _raise_not_found(detail: str = "Resource not found") -> NoReturn:
    """Raise not found exception."""
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=detail,
    )


def _raise_internal_server_error() -> NoReturn:
    """Raise internal server error exception."""
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error",
    )


@router.post(
    "/upload",
    summary="Upload Health Data",
    description="Upload health metrics for processing and analysis by the CLARITY digital twin platform.",
)
@require_auth(permissions=[Permission.WRITE_OWN_DATA])
async def upload_health_data(
    health_data: HealthDataUpload,
    current_user: UserContext = Depends(get_current_user),  # noqa: B008
    service: HealthDataService = Depends(get_health_data_service),  # noqa: B008
) -> HealthDataResponse:
    """Upload health data for processing."""
    try:
        logger.info("Health data upload requested by user: %s", current_user.user_id)

        # Validate user owns the data
        if str(health_data.user_id) != current_user.user_id:
            _raise_forbidden_access()

        # Process health data
        response = await service.process_health_data(health_data)

    except HealthDataServiceError as e:
        logger.exception("Health data service error")
        _raise_service_error(f"Health data processing failed: {e.message}")
    except Exception:
        logger.exception("Unexpected error in health data upload")
        _raise_internal_server_error()
    else:
        logger.info("Health data uploaded successfully: %s", response.processing_id)
        return response


@router.get(
    "/processing/{processing_id}",
    summary="Get Processing Status",
    description="Check the processing status of a health data upload using its processing ID.",
)
@require_auth(permissions=[Permission.READ_OWN_DATA])
async def get_processing_status(
    processing_id: UUID,
    current_user: UserContext = Depends(get_current_user),  # noqa: B008
    service: HealthDataService = Depends(get_health_data_service),  # noqa: B008
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
            _raise_not_found("Processing job not found")

    except HealthDataServiceError as e:
        logger.exception("Health data service error")
        _raise_service_error(str(e))
    except Exception:
        logger.exception("Unexpected error getting processing status")
        _raise_internal_server_error()
    else:
        return status_info


@router.get("/health-data")
@require_auth(permissions=[Permission.READ_OWN_DATA])
async def get_health_data(
    current_user: UserContext = Depends(get_current_user),  # noqa: B008
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    metric_type: str | None = Query(None, description="Filter by metric type"),
    start_date: datetime | None = Query(None, description="Filter from date"),
    end_date: datetime | None = Query(None, description="Filter to date"),  # noqa: B008
    service: HealthDataService = Depends(get_health_data_service),  # noqa: B008
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

    except HealthDataServiceError as e:
        logger.exception("Health data service error")
        _raise_service_error(str(e))
    except Exception:
        logger.exception("Unexpected error retrieving health data")
        _raise_internal_server_error()
    else:
        logger.info(
            "Retrieved %s health data records for user: %s",
            len(health_data.get("metrics", [])),
            current_user.user_id,
        )
        return health_data


@router.delete("/health-data/{processing_id}")
@require_auth(permissions=[Permission.WRITE_OWN_DATA])
async def delete_health_data(
    processing_id: UUID,
    current_user: UserContext = Depends(get_current_user),  # noqa: B008
    service: HealthDataService = Depends(get_health_data_service),  # noqa: B008
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
            _raise_not_found("Processing job not found or access denied")

    except HealthDataServiceError as e:
        logger.exception("Health data service error")
        _raise_service_error(str(e))
    except Exception:
        logger.exception("Unexpected error deleting health data")
        _raise_internal_server_error()
    else:
        logger.info("Health data deleted successfully: %s", processing_id)
        return {"message": "Health data deleted successfully"}


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint for the health data service."""
    try:
        # Basic health check - don't depend on service dependencies
        # This should work even if dependencies aren't injected
        return {
            "status": "healthy",
            "service": "health-data-api",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception("Health check failed")
        return {
            "status": "unhealthy",
            "service": "health-data-api",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }
