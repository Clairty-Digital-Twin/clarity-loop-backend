"""HealthKit Upload API Endpoint.

FastAPI router for handling HealthKit data uploads with immediate acknowledgment
and asynchronous processing via Pub/Sub.
"""

# removed - breaks FastAPI

from datetime import UTC, datetime
import logging
import os
from typing import Any
import uuid

from fastapi import APIRouter, HTTPException, status
from google.cloud import storage
from pydantic import BaseModel, Field

from clarity.auth.dependencies import AuthenticatedUser
from clarity.services.messaging.publisher import get_publisher

# Configure logger
logger = logging.getLogger(__name__)

# Configure router
router = APIRouter(tags=["healthkit"])

# Constants for upload ID validation
MIN_UPLOAD_ID_PARTS = 2
UUID_HEX_LENGTH = 32


def _raise_user_mismatch_error() -> None:
    """Raise HTTPException for user mismatch."""
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Cannot upload data for a different user",
    )


def _raise_invalid_format_error() -> None:
    """Raise ValueError for invalid upload ID format."""
    msg = "Invalid format"
    raise ValueError(msg)


def _raise_invalid_uuid_error() -> None:
    """Raise ValueError for invalid UUID format."""
    msg = "Invalid UUID format"
    raise ValueError(msg)


class HealthKitSample(BaseModel):
    """Individual HealthKit sample."""

    identifier: str
    type: str
    value: float | dict[str, Any]
    unit: str | None = None
    start_date: str
    end_date: str
    source_name: str | None = None
    device: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthKitUploadRequest(BaseModel):
    """HealthKit data upload request."""

    user_id: str
    quantity_samples: list[HealthKitSample] = Field(default_factory=list)
    category_samples: list[HealthKitSample] = Field(default_factory=list)
    workouts: list[dict[str, Any]] = Field(default_factory=list)
    correlation_samples: list[dict[str, Any]] = Field(default_factory=list)
    upload_metadata: dict[str, Any] = Field(default_factory=dict)
    sync_token: str | None = None


class HealthKitUploadResponse(BaseModel):
    """Response for HealthKit upload."""

    upload_id: str
    status: str
    queued_at: str
    samples_received: dict[str, int]
    message: str


@router.post(
    "/",
    response_model=HealthKitUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload HealthKit Data",
    description="Upload HealthKit data for asynchronous processing",
)
@router.post(
    "",
    response_model=HealthKitUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    include_in_schema=False,  # Don't show duplicate in OpenAPI docs
)
async def upload_healthkit_data(
    request: HealthKitUploadRequest,
    current_user: AuthenticatedUser,
) -> HealthKitUploadResponse:
    """Upload HealthKit data for asynchronous processing.

    This endpoint:
    1. Authenticates the request via AWS Cognito
    2. Validates user authorization
    3. Stores raw data to GCS
    4. Publishes Pub/Sub message for processing
    5. Returns immediate acknowledgment

    Args:
        request: HealthKit upload data
        current_user: Authenticated user context from AWS Cognito

    Returns:
        Upload acknowledgment with tracking ID

    Raises:
        HTTPException: For authentication/authorization failures
    """
    try:
        # 1. Authorize user access
        if current_user.user_id != request.user_id:
            _raise_user_mismatch_error()

        # 2. Generate unique upload ID
        upload_id = f"{request.user_id}-{uuid.uuid4().hex}"

        # 3. Save raw data to GCS
        storage_client = storage.Client()
        bucket_name = os.getenv("HEALTHKIT_RAW_BUCKET", "clarity-healthkit-raw")
        bucket = storage_client.bucket(bucket_name)

        # Create hierarchical path for organization
        timestamp = datetime.now(UTC).strftime("%Y/%m/%d")
        blob_path = f"uploads/{timestamp}/{request.user_id}/{upload_id}.json"
        blob = bucket.blob(blob_path)

        # Store the entire request as JSON
        upload_data = request.model_dump()
        upload_data["upload_timestamp"] = datetime.now(UTC).isoformat()
        upload_data["upload_id"] = upload_id

        blob.upload_from_string(
            request.model_dump_json(indent=2), content_type="application/json"
        )

        # 4. Publish to Pub/Sub for processing
        publisher = await get_publisher()

        await publisher.publish_health_data_upload(
            user_id=request.user_id,
            upload_id=upload_id,
            s3_path=f"gs://{bucket_name}/{blob_path}",
            metadata={
                "sample_counts": {
                    "quantity": len(request.quantity_samples),
                    "category": len(request.category_samples),
                    "workouts": len(request.workouts),
                    "correlations": len(request.correlation_samples),
                },
                "sync_token": request.sync_token,
                "upload_source": "healthkit_ios",
            },
        )

        # 5. Return immediate acknowledgment
        return HealthKitUploadResponse(
            upload_id=upload_id,
            status="queued",
            queued_at=datetime.now(UTC).isoformat(),
            samples_received={
                "quantity_samples": len(request.quantity_samples),
                "category_samples": len(request.category_samples),
                "workouts": len(request.workouts),
                "correlation_samples": len(request.correlation_samples),
            },
            message="Health data queued for processing successfully",
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to process HealthKit upload")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process health data upload",
        ) from None


@router.get("/status/{upload_id}")
async def get_upload_status(
    upload_id: str,
    current_user: AuthenticatedUser,
) -> dict[str, Any]:
    """Get status of a HealthKit upload.

    Args:
        upload_id: The upload ID to check
        current_user: Authenticated user context from AWS Cognito

    Returns:
        Upload status information
    """
    # Extract user_id from upload_id (format: user_id-uuid)
    # The UUID is always the last 32-character hex string after the final dash
    try:
        parts = upload_id.split("-")
        if len(parts) < MIN_UPLOAD_ID_PARTS:
            _raise_invalid_format_error()
        # The last part should be a 32-character hex UUID
        uuid_part = parts[-1]
        if len(uuid_part) != UUID_HEX_LENGTH or not all(
            c in "0123456789abcdef" for c in uuid_part.lower()
        ):
            _raise_invalid_uuid_error()
        # Join all parts except the last one to get the user_id
        user_id = "-".join(parts[:-1])
    except (IndexError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid upload ID format"
        ) from None

    # Verify user access
    if current_user.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this upload"
        )

    # TODO: Implement status checking from DynamoDB
    # For now, return a mock response
    return {
        "upload_id": upload_id,
        "status": "processing",
        "progress": 0.75,
        "message": "Analyzing cardiovascular patterns",
        "last_updated": datetime.now(UTC).isoformat(),
    }


# Note: logger already configured at top of file
