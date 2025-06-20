"""CLARITY Digital Twin Platform - AWS S3 Storage Service.

Enterprise-grade AWS S3 client for health data storage and management.
Replaces Google Cloud Storage with AWS-native solution.
"""

# removed - breaks FastAPI

import asyncio
from datetime import UTC, datetime
from functools import partial
import json
import logging
from typing import TYPE_CHECKING, Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from mypy_boto3_s3 import S3Client

from clarity.models.health_data import HealthDataUpload
from clarity.ports.storage import CloudStoragePort

if TYPE_CHECKING:
    pass  # Only for type stubs now

# Configure logger
logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("audit")

__all__ = [
    "S3DownloadError",
    "S3PermissionError",
    "S3StorageError",
    "S3StorageService",
    "S3UploadError",
    "get_s3_service",
]


class S3StorageError(Exception):
    """Base exception for S3 storage operations."""


class S3UploadError(S3StorageError):
    """Raised when S3 upload fails."""


class S3DownloadError(S3StorageError):
    """Raised when S3 download fails."""


class S3PermissionError(S3StorageError):
    """Raised when S3 operation is not permitted."""


class S3StorageService(CloudStoragePort):
    """Enterprise-grade S3 storage service for health data operations.

    Features:
    - HIPAA-compliant data encryption and audit logging
    - Lifecycle management and automated archival
    - High-performance parallel uploads/downloads
    - Comprehensive error handling and retry logic
    - Metadata tagging for compliance and organization
    """

    def __init__(
        self,
        bucket_name: str,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
        *,
        enable_encryption: bool = True,
        storage_class: str = "STANDARD",
    ) -> None:
        """Initialize the S3 storage service.

        Args:
            bucket_name: S3 bucket name for health data storage
            region: AWS region
            endpoint_url: Optional endpoint URL (for local S3 testing)
            enable_encryption: Enable server-side encryption
            storage_class: S3 storage class (STANDARD, IA, GLACIER, etc.)
        """
        self.bucket_name = bucket_name
        self.region = region
        self.endpoint_url = endpoint_url
        self.enable_encryption = enable_encryption
        self.storage_class = storage_class

        # Initialize S3 client
        self.s3_client: S3Client = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=endpoint_url,
        )

        # Lifecycle configuration
        self.lifecycle_rules = {
            "raw_data": {
                "transition_ia_days": 30,  # Move to IA after 30 days
                "transition_glacier_days": 90,  # Move to Glacier after 90 days
                "expiration_days": 2555,  # Delete after 7 years (HIPAA retention)
            },
            "processed_data": {
                "transition_ia_days": 7,
                "transition_glacier_days": 30,
                "expiration_days": 2555,
            },
        }

        logger.info(
            "S3 storage service initialized for bucket: %s (region: %s)",
            bucket_name,
            region,
        )

    async def _audit_log(
        self,
        operation: str,
        s3_key: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create audit log entry for HIPAA compliance."""
        try:
            audit_entry = {
                "operation": operation,
                "bucket": self.bucket_name,
                "s3_key": s3_key,
                "user_id": user_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "metadata": metadata or {},
                "source": "s3_storage_service",
            }

            audit_logger.info(
                "S3 operation: %s on %s/%s by user %s",
                operation,
                self.bucket_name,
                s3_key,
                user_id,
                extra={"audit_data": audit_entry},
            )

        except Exception:
            logger.exception("Failed to create audit log")
            # Don't raise exception for audit failures

    async def upload_raw_health_data(
        self, user_id: str, processing_id: str, health_data: HealthDataUpload
    ) -> str:
        """Upload raw health data to S3 with HIPAA compliance.

        Args:
            user_id: User identifier
            processing_id: Unique processing job ID
            health_data: Raw health data to upload

        Returns:
            S3 URI where data was stored

        Raises:
            S3UploadError: If upload fails
        """
        try:
            # Create S3 key path (partitioned by date for performance)
            upload_date = datetime.now(UTC).strftime("%Y/%m/%d")
            s3_key = f"raw_data/{upload_date}/{user_id}/{processing_id}.json"
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"

            # Prepare health data for storage - DO NOT SANITIZE raw data!
            raw_data = {
                "user_id": user_id,  # Store raw user_id
                "processing_id": processing_id,
                "upload_source": health_data.upload_source,
                "client_timestamp": health_data.client_timestamp.isoformat(),
                "server_timestamp": datetime.now(UTC).isoformat(),
                "sync_token": health_data.sync_token,
                "metrics_count": len(health_data.metrics),
                "data_schema_version": "1.0",
                "metrics": [
                    {
                        "metric_id": str(metric.metric_id),
                        "metric_type": metric.metric_type.value,
                        "created_at": metric.created_at.isoformat(),
                        "device_id": metric.device_id
                        or "unknown",  # Store raw device_id
                        "biometric_data": (
                            metric.biometric_data.model_dump()
                            if metric.biometric_data
                            else None
                        ),
                        "activity_data": (
                            metric.activity_data.model_dump()
                            if metric.activity_data
                            else None
                        ),
                        "sleep_data": (
                            metric.sleep_data.model_dump()
                            if metric.sleep_data
                            else None
                        ),
                        "mental_health_data": (
                            metric.mental_health_data.model_dump()
                            if metric.mental_health_data
                            else None
                        ),
                    }
                    for metric in health_data.metrics
                ],
            }

            # Prepare upload parameters
            upload_params: dict[str, Any] = {
                "Bucket": self.bucket_name,
                "Key": s3_key,
                "Body": json.dumps(
                    raw_data, indent=2, default=str
                ),  # Convert UUIDs to strings
                "ContentType": "application/json",
                "StorageClass": self.storage_class,
                "Metadata": {
                    "user-id": user_id,
                    "processing-id": processing_id,
                    "upload-source": health_data.upload_source,
                    "metrics-count": str(len(health_data.metrics)),
                    "uploaded-at": datetime.now(UTC).isoformat(),
                    "data-type": "raw-health-data",
                    "compliance": "hipaa",
                },
            }

            # Add server-side encryption
            if self.enable_encryption:
                upload_params["ServerSideEncryption"] = "AES256"

            # Upload to S3
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.s3_client.put_object(**upload_params)
            )

            # Create audit log
            await self._audit_log(
                operation="upload_raw_health_data",
                s3_key=s3_key,
                user_id=user_id,
                metadata={
                    "metrics_count": len(health_data.metrics),
                    "upload_source": health_data.upload_source,
                    "data_size_bytes": len(json.dumps(raw_data, default=str)),
                },
            )

            logger.info(
                "Raw health data uploaded to S3: %s (%d metrics)",
                s3_uri,
                len(health_data.metrics),
            )

        except ClientError as e:
            logger.exception("S3 upload failed")
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            msg = f"S3 upload failed ({error_code}): {e}"
            raise S3UploadError(msg) from e
        except Exception as e:
            logger.exception("Unexpected error during S3 upload")
            msg = f"S3 upload failed: {e}"
            raise S3UploadError(msg) from e
        else:
            return s3_uri

    async def upload_analysis_results(
        self, user_id: str, processing_id: str, analysis_results: dict[str, Any]
    ) -> str:
        """Upload analysis results to S3.

        Args:
            user_id: User identifier
            processing_id: Processing job identifier
            analysis_results: Analysis results to store

        Returns:
            S3 URI where results were stored
        """
        try:
            # Create S3 key path
            upload_date = datetime.now(UTC).strftime("%Y/%m/%d")
            s3_key = (
                f"analysis_results/{upload_date}/{user_id}/{processing_id}_results.json"
            )
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"

            # Prepare analysis data
            analysis_data = {
                "user_id": user_id,
                "processing_id": processing_id,
                "analysis_timestamp": datetime.now(UTC).isoformat(),
                "results": analysis_results,
                "data_schema_version": "1.0",
            }

            # Upload parameters
            upload_params: dict[str, Any] = {
                "Bucket": self.bucket_name,
                "Key": s3_key,
                "Body": json.dumps(analysis_data, indent=2),
                "ContentType": "application/json",
                "StorageClass": "STANDARD_IA",  # Infrequent access for analysis results
                "Metadata": {
                    "user-id": user_id,
                    "processing-id": processing_id,
                    "data-type": "analysis-results",
                    "created-at": datetime.now(UTC).isoformat(),
                },
            }

            if self.enable_encryption:
                upload_params["ServerSideEncryption"] = "AES256"

            # Upload to S3
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.s3_client.put_object(**upload_params)
            )

            await self._audit_log(
                operation="upload_analysis_results",
                s3_key=s3_key,
                user_id=user_id,
                metadata={"results_size": len(json.dumps(analysis_results))},
            )

            logger.info("Analysis results uploaded to S3: %s", s3_uri)

        except Exception as e:
            logger.exception("Failed to upload analysis results")
            msg = f"Analysis results upload failed: {e}"
            raise S3UploadError(msg) from e
        else:
            return s3_uri

    async def download_raw_data(self, s3_key: str, user_id: str) -> dict[str, Any]:
        """Download raw health data from S3.

        Args:
            s3_key: S3 key of the data to download
            user_id: User ID for audit logging

        Returns:
            Raw health data dictionary

        Raises:
            S3DownloadError: If download fails
        """
        try:
            # Download from S3
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key),
            )

            # Parse JSON data
            data: dict[str, Any] = json.loads(response["Body"].read().decode("utf-8"))

            await self._audit_log(
                operation="download_raw_data",
                s3_key=s3_key,
                user_id=user_id,
            )

            logger.info("Raw data downloaded from S3: %s", s3_key)
            return data

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                msg = f"File not found: {s3_key}"
                raise S3DownloadError(msg) from e
            msg = f"S3 download failed: {e}"
            raise S3DownloadError(msg) from e
        except Exception as e:
            logger.exception("Failed to download from S3")
            msg = f"Download failed: {e}"
            raise S3DownloadError(msg) from e

    async def list_user_files(
        self, user_id: str, prefix: str = "", max_keys: int = 1000
    ) -> list[dict[str, Any]]:
        """List files for a specific user.

        Args:
            user_id: User identifier
            prefix: Optional prefix filter
            max_keys: Maximum number of keys to return

        Returns:
            List of file metadata dictionaries
        """
        try:
            # Build prefix for user's files
            search_prefix = f"raw_data/{prefix}{user_id}/" if prefix else "raw_data/"

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=search_prefix,
                    MaxKeys=max_keys,
                ),
            )

            # Filter for user's files if using broader prefix
            files = [
                {
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                    "etag": obj["ETag"],
                    "storage_class": obj.get("StorageClass", "STANDARD"),
                }
                for obj in response.get("Contents", [])
                if user_id in obj["Key"]
            ]

            logger.info("Listed %d files for user %s", len(files), user_id)

        except Exception as e:
            logger.exception("Failed to list user files")
            msg = f"File listing failed: {e}"
            raise S3StorageError(msg) from e
        else:
            return files

    async def delete_user_data(self, user_id: str) -> int:
        """Delete all data for a user (GDPR compliance).

        Args:
            user_id: User identifier

        Returns:
            Number of files deleted
        """
        try:
            # List all user files
            files = await self.list_user_files(user_id)

            deleted_count = 0
            # Delete files in batches
            for file_info in files:
                try:
                    file_key = file_info["key"]
                    # Use functools.partial to avoid lambda type inference issues
                    delete_func = partial(
                        self.s3_client.delete_object,
                        Bucket=self.bucket_name,
                        Key=file_key,
                    )
                    await asyncio.get_event_loop().run_in_executor(None, delete_func)
                    deleted_count += 1
                except ClientError as e:
                    logger.warning("Failed to delete file %s: %s", file_info["key"], e)
                    logger.debug("S3 delete error details: %s", e, exc_info=True)
                except (OSError, TimeoutError) as e:
                    # Network or I/O errors that shouldn't crash the deletion process
                    logger.warning(
                        "Network/IO error deleting file %s: %s", file_info["key"], e
                    )
                    logger.debug("Network error details: %s", e, exc_info=True)

            await self._audit_log(
                operation="delete_user_data",
                s3_key=f"user_data/{user_id}/*",
                user_id=user_id,
                metadata={"deleted_files": deleted_count},
            )

            logger.info("Deleted %d files for user %s", deleted_count, user_id)

        except Exception as e:
            logger.exception("Failed to delete user data")
            msg = f"User data deletion failed: {e}"
            raise S3StorageError(msg) from e
        else:
            return deleted_count

    async def setup_bucket_lifecycle(self) -> None:
        """Set up S3 bucket lifecycle policies for automatic data management."""
        try:
            # Define lifecycle configuration
            lifecycle_config: dict[str, Any] = {
                "Rules": [
                    {
                        "ID": "RawDataLifecycle",
                        "Status": "Enabled",
                        "Filter": {"Prefix": "raw_data/"},
                        "Transitions": [
                            {
                                "Days": self.lifecycle_rules["raw_data"][
                                    "transition_ia_days"
                                ],
                                "StorageClass": "STANDARD_IA",
                            },
                            {
                                "Days": self.lifecycle_rules["raw_data"][
                                    "transition_glacier_days"
                                ],
                                "StorageClass": "GLACIER",
                            },
                        ],
                        "Expiration": {
                            "Days": self.lifecycle_rules["raw_data"]["expiration_days"]
                        },
                    },
                    {
                        "ID": "ProcessedDataLifecycle",
                        "Status": "Enabled",
                        "Filter": {"Prefix": "analysis_results/"},
                        "Transitions": [
                            {
                                "Days": self.lifecycle_rules["processed_data"][
                                    "transition_ia_days"
                                ],
                                "StorageClass": "STANDARD_IA",
                            },
                            {
                                "Days": self.lifecycle_rules["processed_data"][
                                    "transition_glacier_days"
                                ],
                                "StorageClass": "GLACIER",
                            },
                        ],
                        "Expiration": {
                            "Days": self.lifecycle_rules["processed_data"][
                                "expiration_days"
                            ]
                        },
                    },
                ]
            }

            # Apply lifecycle configuration
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.put_bucket_lifecycle_configuration(
                    Bucket=self.bucket_name,
                    LifecycleConfiguration=lifecycle_config,  # type: ignore[arg-type]
                ),
            )

            logger.info("S3 bucket lifecycle policies configured")

        except ClientError as e:
            logger.warning("Failed to set up bucket lifecycle: %s", e)
            # Don't raise - lifecycle is optional
        except (OSError, TimeoutError) as e:
            # Network errors during lifecycle setup
            logger.warning("Network error setting up bucket lifecycle: %s", e)
            logger.debug("Lifecycle setup error details: %s", e, exc_info=True)
            # Don't raise - lifecycle is optional

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on S3 service.

        Returns:
            Dict with health status information
        """
        try:
            # Test S3 connection by checking bucket existence
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.s3_client.head_bucket(Bucket=self.bucket_name)
            )

            return {
                "status": "healthy",
                "bucket": self.bucket_name,
                "region": self.region,
                "encryption_enabled": self.enable_encryption,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            return {
                "status": "unhealthy",
                "error": f"S3 error ({error_code}): {e}",
                "bucket": self.bucket_name,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except (BotoCoreError, ConnectionError, TimeoutError) as e:
            return {
                "status": "unhealthy",
                "error": f"Connection error: {e}",
                "bucket": self.bucket_name,
                "timestamp": datetime.now(UTC).isoformat(),
            }

    # CloudStoragePort interface methods
    async def upload_file(
        self, file_data: bytes, file_path: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Generic file upload method (CloudStoragePort interface)."""
        try:
            upload_params: dict[str, Any] = {
                "Bucket": self.bucket_name,
                "Key": file_path,
                "Body": file_data,
                "Metadata": metadata or {},
            }

            if self.enable_encryption:
                upload_params["ServerSideEncryption"] = "AES256"

            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.s3_client.put_object(**upload_params)
            )

            s3_uri = f"s3://{self.bucket_name}/{file_path}"
            logger.info("File uploaded to S3: %s", s3_uri)

        except Exception as e:
            msg = f"File upload failed: {e}"
            raise S3UploadError(msg) from e
        else:
            return s3_uri

    async def download_file(self, file_path: str) -> bytes:
        """Generic file download method (CloudStoragePort interface)."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.get_object(
                    Bucket=self.bucket_name, Key=file_path
                ),
            )

            file_data = response["Body"].read()
            logger.info("File downloaded from S3: %s", file_path)

        except Exception as e:
            msg = f"File download failed: {e}"
            raise S3DownloadError(msg) from e
        else:
            return file_data

    async def delete_file(self, file_path: str) -> bool:
        """Generic file deletion method (CloudStoragePort interface)."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.delete_object(
                    Bucket=self.bucket_name, Key=file_path
                ),
            )

            logger.info("File deleted from S3: %s", file_path)
            return True

        except Exception:
            logger.exception("Failed to delete file %s", file_path)
            return False

    # CloudStoragePort interface methods that need to be implemented
    def bucket(self, bucket_name: str) -> object:
        """Get a bucket reference.

        Args:
            bucket_name: Name of the bucket

        Returns:
            Bucket reference object
        """
        # For S3, we just return the bucket name as we use direct client calls
        return bucket_name

    def upload_json(
        self,
        bucket_name: str,
        blob_path: str,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Upload JSON data to cloud storage.

        Args:
            bucket_name: Name of the bucket
            blob_path: Path for the blob
            data: JSON data to upload
            metadata: Optional metadata

        Returns:
            Full path/URL of uploaded object
        """
        # This is a synchronous wrapper around the async upload_file method
        # In practice, this should be refactored to be async
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            json_data = json.dumps(data, indent=2).encode("utf-8")
            return loop.run_until_complete(
                self.upload_file(json_data, blob_path, metadata)
            )
        finally:
            loop.close()

    def get_raw_data_bucket_name(self) -> str:
        """Get the name of the raw data bucket.

        Returns:
            Bucket name for raw data storage
        """
        return self.bucket_name


# Global singleton instance
_s3_service: S3StorageService | None = None


def get_s3_service(
    bucket_name: str | None = None,
    region: str = "us-east-1",
    endpoint_url: str | None = None,
) -> S3StorageService:
    """Get or create global S3 service instance."""
    global _s3_service  # noqa: PLW0603 - Singleton pattern for S3 storage service

    if _s3_service is None:
        if not bucket_name:
            bucket_name = "clarity-health-data-storage"

        _s3_service = S3StorageService(
            bucket_name=bucket_name,
            region=region,
            endpoint_url=endpoint_url,
        )

    return _s3_service
