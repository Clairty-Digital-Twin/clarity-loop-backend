"""Comprehensive tests for S3 storage service."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
from typing import Any
from unittest.mock import MagicMock, patch
import uuid

from botocore.exceptions import BotoCoreError, ClientError
import pytest

from clarity.models.health_data import (
    ActivityData,
    BiometricData,
    HealthDataUpload,
    HealthMetric,
    HealthMetricType,
)
from clarity.services.s3_storage_service import (
    S3DownloadError,
    S3PermissionError,
    S3StorageError,
    S3StorageService,
    S3UploadError,
    get_s3_service,
)


@pytest.fixture
def mock_s3_client() -> MagicMock:
    """Mock S3 client."""
    with patch("boto3.client") as mock_client:
        yield mock_client.return_value


@pytest.fixture
def s3_service(mock_s3_client: MagicMock) -> S3StorageService:
    """Create S3 storage service with mocked client."""
    service = S3StorageService(
        bucket_name="test-health-bucket",
        region="us-east-1",
        enable_encryption=True,
        storage_class="STANDARD",
    )
    service.s3_client = mock_s3_client
    return service


@pytest.fixture
def valid_health_data() -> HealthDataUpload:
    """Create valid health data upload."""
    return HealthDataUpload(
        user_id=str(uuid.uuid4()),
        upload_source="mobile_app",
        client_timestamp=datetime.now(UTC),
        sync_token="sync-token-123",  # noqa: S106 - Test sync token value
        metrics=[
            HealthMetric(
                metric_id=uuid.uuid4(),
                metric_type=HealthMetricType.HEART_RATE,
                created_at=datetime.now(UTC),
                device_id="device-123",
                biometric_data=BiometricData(
                    heart_rate=72,
                    systolic_bp=120,
                    diastolic_bp=80,
                    temperature=98.6,
                    oxygen_saturation=98,
                ),
            ),
            HealthMetric(
                metric_id=uuid.uuid4(),
                metric_type=HealthMetricType.ACTIVITY_LEVEL,
                created_at=datetime.now(UTC),
                device_id="device-123",
                activity_data=ActivityData(
                    steps=5000,
                    distance=3.2,
                    calories_burned=250,
                    active_minutes=45,
                    activity_type="walking",
                ),
            ),
        ],
    )


class TestS3StorageServiceInit:
    """Test S3 storage service initialization."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        with patch("boto3.client") as mock_boto_client:
            service = S3StorageService(bucket_name="test-bucket")

            assert service.bucket_name == "test-bucket"
            assert service.region == "us-east-1"
            assert service.endpoint_url is None
            assert service.enable_encryption is True
            assert service.storage_class == "STANDARD"

            mock_boto_client.assert_called_once_with(
                "s3",
                region_name="us-east-1",
                endpoint_url=None,
            )

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        with patch("boto3.client") as mock_boto_client:
            service = S3StorageService(
                bucket_name="custom-bucket",
                region="eu-west-1",
                endpoint_url="http://localhost:9000",
                enable_encryption=False,
                storage_class="GLACIER",
            )

            assert service.bucket_name == "custom-bucket"
            assert service.region == "eu-west-1"
            assert service.endpoint_url == "http://localhost:9000"
            assert service.enable_encryption is False
            assert service.storage_class == "GLACIER"

            mock_boto_client.assert_called_once_with(
                "s3",
                region_name="eu-west-1",
                endpoint_url="http://localhost:9000",
            )

    def test_lifecycle_rules(self, s3_service: S3StorageService) -> None:
        """Test lifecycle rules configuration."""
        assert "raw_data" in s3_service.lifecycle_rules
        assert "processed_data" in s3_service.lifecycle_rules

        raw_rules = s3_service.lifecycle_rules["raw_data"]
        assert raw_rules["transition_ia_days"] == 30
        assert raw_rules["transition_glacier_days"] == 90
        assert raw_rules["expiration_days"] == 2555


class TestAuditLog:
    """Test audit logging functionality."""

    @pytest.mark.asyncio
    async def test_audit_log_success(self, s3_service: S3StorageService) -> None:
        """Test successful audit log creation."""
        with patch("clarity.services.s3_storage_service.audit_logger") as mock_logger:
            await s3_service._audit_log(
                operation="test_operation",
                s3_key="test/key.json",
                user_id="user-123",
                metadata={"size": 1024},
            )

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args

            # Check the format string and arguments
            assert call_args[0][0] == "S3 operation: %s on %s/%s by user %s"
            assert call_args[0][1] == "test_operation"
            assert call_args[0][2] == "test-health-bucket"
            assert call_args[0][3] == "test/key.json"
            assert call_args[0][4] == "user-123"

            extra_data = call_args[1]["extra"]["audit_data"]
            assert extra_data["operation"] == "test_operation"
            assert extra_data["bucket"] == "test-health-bucket"
            assert extra_data["s3_key"] == "test/key.json"
            assert extra_data["user_id"] == "user-123"
            assert extra_data["metadata"] == {"size": 1024}

    @pytest.mark.asyncio
    async def test_audit_log_failure(self, s3_service: S3StorageService) -> None:
        """Test audit log creation failure (should not raise)."""
        with patch("clarity.services.s3_storage_service.audit_logger") as mock_logger:
            mock_logger.info.side_effect = Exception("Logger error")

            # Should not raise exception
            await s3_service._audit_log(
                operation="test_operation",
                s3_key="test/key.json",
            )

    @pytest.mark.asyncio
    async def test_audit_log_no_metadata(self, s3_service: S3StorageService) -> None:
        """Test audit log with no metadata."""
        with patch("clarity.services.s3_storage_service.audit_logger") as mock_logger:
            await s3_service._audit_log(
                operation="test_operation",
                s3_key="test/key.json",
                user_id="test-user",
            )

            mock_logger.info.assert_called_once()
            extra_data = mock_logger.info.call_args[1]["extra"]["audit_data"]
            assert extra_data["metadata"] == {}  # Should be empty dict


class TestUploadRawHealthData:
    """Test raw health data upload functionality."""

    @pytest.mark.asyncio
    async def test_upload_raw_health_data_success(
        self,
        s3_service: S3StorageService,
        mock_s3_client: MagicMock,
        valid_health_data: HealthDataUpload,
    ) -> None:
        """Test successful health data upload."""
        processing_id = "proc-123"
        user_id = valid_health_data.user_id

        with patch("clarity.services.s3_storage_service.datetime") as mock_datetime:
            mock_date = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
            mock_datetime.now.return_value = mock_date

            s3_uri = await s3_service.upload_raw_health_data(
                user_id, processing_id, valid_health_data
            )

        expected_key = f"raw_data/2024/01/15/{user_id}/{processing_id}.json"
        assert s3_uri == f"s3://test-health-bucket/{expected_key}"

        # Verify put_object was called
        mock_s3_client.put_object.assert_called_once()
        call_kwargs = mock_s3_client.put_object.call_args[1]

        assert call_kwargs["Bucket"] == "test-health-bucket"
        assert call_kwargs["Key"] == expected_key
        assert call_kwargs["ContentType"] == "application/json"
        assert call_kwargs["StorageClass"] == "STANDARD"
        assert call_kwargs["ServerSideEncryption"] == "AES256"

        # Verify metadata
        metadata = call_kwargs["Metadata"]
        assert metadata["user-id"] == user_id
        assert metadata["processing-id"] == processing_id
        assert metadata["upload-source"] == "mobile_app"
        assert metadata["metrics-count"] == "2"
        assert metadata["data-type"] == "raw-health-data"
        assert metadata["compliance"] == "hipaa"

        # Verify body content
        body_data = json.loads(call_kwargs["Body"])
        assert body_data["user_id"] == str(
            user_id
        )  # Convert UUID to string for comparison
        assert body_data["processing_id"] == processing_id
        assert body_data["metrics_count"] == 2
        assert len(body_data["metrics"]) == 2

    @pytest.mark.asyncio
    async def test_upload_raw_health_data_no_encryption(
        self, mock_s3_client: MagicMock, valid_health_data: HealthDataUpload
    ) -> None:
        """Test upload without encryption."""
        service = S3StorageService(
            bucket_name="test-bucket",
            enable_encryption=False,
        )
        service.s3_client = mock_s3_client

        await service.upload_raw_health_data("user-123", "proc-123", valid_health_data)

        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert "ServerSideEncryption" not in call_kwargs

    @pytest.mark.asyncio
    async def test_upload_raw_health_data_client_error(
        self,
        s3_service: S3StorageService,
        mock_s3_client: MagicMock,
        valid_health_data: HealthDataUpload,
    ) -> None:
        """Test upload with ClientError."""
        mock_s3_client.put_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access denied"}},
            "PutObject",
        )

        with pytest.raises(S3UploadError) as exc_info:
            await s3_service.upload_raw_health_data(
                "user-123", "proc-123", valid_health_data
            )

        assert "S3 upload failed (AccessDenied)" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_upload_raw_health_data_unexpected_error(
        self,
        s3_service: S3StorageService,
        mock_s3_client: MagicMock,
        valid_health_data: HealthDataUpload,
    ) -> None:
        """Test upload with unexpected error."""
        mock_s3_client.put_object.side_effect = Exception("Unexpected error")

        with pytest.raises(S3UploadError) as exc_info:
            await s3_service.upload_raw_health_data(
                "user-123", "proc-123", valid_health_data
            )

        assert "S3 upload failed" in str(exc_info.value)


class TestUploadAnalysisResults:
    """Test analysis results upload functionality."""

    @pytest.mark.asyncio
    async def test_upload_analysis_results_success(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test successful analysis results upload."""
        user_id = "user-123"
        processing_id = "proc-456"
        analysis_results = {
            "summary": {"risk_score": 0.2, "health_status": "good"},
            "metrics": {"heart_rate_avg": 72, "steps_total": 8500},
        }

        mock_date = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)

        with patch("clarity.services.s3_storage_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_date

            s3_uri = await s3_service.upload_analysis_results(
                user_id, processing_id, analysis_results
            )

        expected_key = (
            f"analysis_results/2024/01/15/{user_id}/{processing_id}_results.json"
        )
        assert s3_uri == f"s3://test-health-bucket/{expected_key}"

        # Verify put_object was called
        mock_s3_client.put_object.assert_called_once()
        call_kwargs = mock_s3_client.put_object.call_args[1]

        assert call_kwargs["Bucket"] == "test-health-bucket"
        assert call_kwargs["Key"] == expected_key
        assert call_kwargs["ContentType"] == "application/json"
        assert call_kwargs["StorageClass"] == "STANDARD_IA"
        assert call_kwargs["ServerSideEncryption"] == "AES256"

        # Verify metadata
        metadata = call_kwargs["Metadata"]
        assert metadata["user-id"] == user_id
        assert metadata["processing-id"] == processing_id
        assert metadata["data-type"] == "analysis-results"

        # Verify body content
        body_data = json.loads(call_kwargs["Body"])
        assert body_data["user_id"] == str(
            user_id
        )  # Convert UUID to string for comparison
        assert body_data["processing_id"] == processing_id
        assert body_data["results"] == analysis_results

    @pytest.mark.asyncio
    async def test_upload_analysis_results_error(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test analysis results upload with error."""
        mock_s3_client.put_object.side_effect = Exception("Upload error")

        with pytest.raises(S3UploadError) as exc_info:
            await s3_service.upload_analysis_results(
                "user-123", "proc-123", {"test": "data"}
            )

        assert "Analysis results upload failed" in str(exc_info.value)


class TestDownloadRawData:
    """Test raw data download functionality."""

    @pytest.mark.asyncio
    async def test_download_raw_data_success(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test successful data download."""
        s3_key = "raw_data/2024/01/15/user-123/proc-123.json"
        test_data = {
            "user_id": "user-123",
            "processing_id": "proc-123",
            "metrics": [{"type": "heart_rate", "value": 72}],
        }

        # Mock response
        mock_response = {"Body": MagicMock()}
        mock_response["Body"].read.return_value = json.dumps(test_data).encode()
        mock_s3_client.get_object.return_value = mock_response

        result = await s3_service.download_raw_data(s3_key, "user-123")

        assert result == test_data
        mock_s3_client.get_object.assert_called_once_with(
            Bucket="test-health-bucket",
            Key=s3_key,
        )

    @pytest.mark.asyncio
    async def test_download_raw_data_not_found(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test download when file doesn't exist."""
        mock_s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Key not found"}},
            "GetObject",
        )

        with pytest.raises(S3DownloadError) as exc_info:
            await s3_service.download_raw_data("missing/key.json", "user-123")

        assert "File not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_download_raw_data_client_error(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test download with other client error."""
        mock_s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access denied"}},
            "GetObject",
        )

        with pytest.raises(S3DownloadError) as exc_info:
            await s3_service.download_raw_data("test/key.json", "user-123")

        assert "S3 download failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_download_raw_data_parse_error(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test download with JSON parse error."""
        mock_response = {"Body": MagicMock()}
        mock_response["Body"].read.return_value = b"invalid json"
        mock_s3_client.get_object.return_value = mock_response

        with pytest.raises(S3DownloadError) as exc_info:
            await s3_service.download_raw_data("test/key.json", "user-123")

        assert "Download failed" in str(exc_info.value)


class TestListUserFiles:
    """Test user file listing functionality."""

    @pytest.mark.asyncio
    async def test_list_user_files_success(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test successful file listing."""
        user_id = "user-123"
        mock_response = {
            "Contents": [
                {
                    "Key": f"raw_data/2024/01/15/{user_id}/file1.json",
                    "Size": 1024,
                    "LastModified": datetime.now(UTC),
                    "ETag": '"abc123"',
                    "StorageClass": "STANDARD",
                },
                {
                    "Key": f"raw_data/2024/01/14/{user_id}/file2.json",
                    "Size": 2048,
                    "LastModified": datetime.now(UTC),
                    "ETag": '"def456"',
                    "StorageClass": "STANDARD",
                },
            ]
        }
        mock_s3_client.list_objects_v2.return_value = mock_response

        files = await s3_service.list_user_files(user_id)

        assert len(files) == 2
        assert files[0]["key"] == f"raw_data/2024/01/15/{user_id}/file1.json"
        assert files[0]["size"] == 1024
        assert files[1]["key"] == f"raw_data/2024/01/14/{user_id}/file2.json"
        assert files[1]["size"] == 2048

        mock_s3_client.list_objects_v2.assert_called_once_with(
            Bucket="test-health-bucket",
            Prefix="raw_data/",
            MaxKeys=1000,
        )

    @pytest.mark.asyncio
    async def test_list_user_files_with_prefix(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test file listing with prefix."""
        mock_s3_client.list_objects_v2.return_value = {"Contents": []}

        await s3_service.list_user_files("user-123", prefix="2024/01/")

        mock_s3_client.list_objects_v2.assert_called_once_with(
            Bucket="test-health-bucket",
            Prefix="raw_data/2024/01/user-123/",
            MaxKeys=1000,
        )

    @pytest.mark.asyncio
    async def test_list_user_files_empty(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test file listing with no results."""
        mock_s3_client.list_objects_v2.return_value = {}

        files = await s3_service.list_user_files("user-123")

        assert files == []

    @pytest.mark.asyncio
    async def test_list_user_files_filters_others(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test file listing filters out other users' files."""
        user_id = "user-123"
        mock_response = {
            "Contents": [
                {
                    "Key": f"raw_data/2024/01/15/{user_id}/file1.json",
                    "Size": 1024,
                    "LastModified": datetime.now(UTC),
                    "ETag": '"abc123"',
                    "StorageClass": "STANDARD",
                },
                {
                    "Key": "raw_data/2024/01/15/other-user/file2.json",  # Different user
                    "Size": 2048,
                    "LastModified": datetime.now(UTC),
                    "ETag": '"def456"',
                    "StorageClass": "STANDARD",
                },
                {
                    "Key": f"raw_data/2024/01/14/{user_id}/file3.json",
                    "Size": 512,
                    "LastModified": datetime.now(UTC),
                    "ETag": '"ghi789"',
                    "StorageClass": "STANDARD",
                },
            ]
        }
        mock_s3_client.list_objects_v2.return_value = mock_response

        files = await s3_service.list_user_files(user_id)

        # Should only return files for user-123, not other-user
        assert len(files) == 2
        assert all(user_id in f["key"] for f in files)
        assert files[0]["key"] == f"raw_data/2024/01/15/{user_id}/file1.json"
        assert files[1]["key"] == f"raw_data/2024/01/14/{user_id}/file3.json"

    @pytest.mark.asyncio
    async def test_list_user_files_error(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test file listing with error."""
        mock_s3_client.list_objects_v2.side_effect = Exception("List error")

        with pytest.raises(S3StorageError) as exc_info:
            await s3_service.list_user_files("user-123")

        assert "File listing failed" in str(exc_info.value)


class TestDeleteFile:
    """Test file deletion functionality."""

    @pytest.mark.asyncio
    async def test_delete_file_success(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test successful file deletion."""
        s3_key = "raw_data/2024/01/15/user-123/file.json"

        result = await s3_service.delete_file(s3_key)

        assert result is True
        mock_s3_client.delete_object.assert_called_once_with(
            Bucket="test-health-bucket",
            Key=s3_key,
        )

    @pytest.mark.asyncio
    async def test_delete_file_error(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test deletion with error."""
        mock_s3_client.delete_object.side_effect = Exception("Delete error")

        # Should return False on error
        result = await s3_service.delete_file("test/file.json")

        assert result is False


class TestDeleteUserData:
    """Test user data deletion functionality."""

    @pytest.mark.asyncio
    async def test_delete_user_data_success(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test successful user data deletion."""
        # Mock list_user_files response
        with patch.object(s3_service, "list_user_files") as mock_list:
            mock_list.return_value = [
                {"key": "raw_data/2024/01/15/user-123/file1.json"},
                {"key": "raw_data/2024/01/14/user-123/file2.json"},
            ]

            deleted_count = await s3_service.delete_user_data("user-123")

            assert deleted_count == 2
            assert mock_s3_client.delete_object.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_user_data_partial_failure(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test user data deletion with partial failures."""
        with patch.object(s3_service, "list_user_files") as mock_list:
            mock_list.return_value = [
                {"key": "raw_data/2024/01/15/user-123/file1.json"},
                {"key": "raw_data/2024/01/14/user-123/file2.json"},
            ]

            # First delete succeeds, second fails
            mock_s3_client.delete_object.side_effect = [
                None,
                ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "Delete error"}},
                    "DeleteObject",
                ),
            ]

            deleted_count = await s3_service.delete_user_data("user-123")

            assert deleted_count == 1  # Only one successful deletion


class TestSetupBucketLifecycle:
    """Test bucket lifecycle setup."""

    @pytest.mark.asyncio
    async def test_setup_bucket_lifecycle_success(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test successful lifecycle setup."""
        await s3_service.setup_bucket_lifecycle()

        mock_s3_client.put_bucket_lifecycle_configuration.assert_called_once()
        call_args = mock_s3_client.put_bucket_lifecycle_configuration.call_args[1]

        assert call_args["Bucket"] == "test-health-bucket"
        assert "Rules" in call_args["LifecycleConfiguration"]

        rules = call_args["LifecycleConfiguration"]["Rules"]
        assert len(rules) == 2  # raw_data and processed_data rules

        # Check raw_data rule
        raw_rule = next(r for r in rules if r["ID"] == "RawDataLifecycle")
        assert raw_rule["Status"] == "Enabled"
        assert raw_rule["Filter"]["Prefix"] == "raw_data/"

    @pytest.mark.asyncio
    async def test_setup_bucket_lifecycle_error(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test lifecycle setup with error (should not raise)."""
        mock_s3_client.put_bucket_lifecycle_configuration.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Setup error"}},
            "PutBucketLifecycleConfiguration",
        )

        # Should not raise exception
        await s3_service.setup_bucket_lifecycle()


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_success(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test successful health check."""
        result = await s3_service.health_check()

        assert result["status"] == "healthy"
        assert result["bucket"] == "test-health-bucket"
        assert result["region"] == "us-east-1"
        assert result["encryption_enabled"] is True
        assert "timestamp" in result

        mock_s3_client.head_bucket.assert_called_once_with(Bucket="test-health-bucket")

    @pytest.mark.asyncio
    async def test_health_check_client_error(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test health check with client error."""
        mock_s3_client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "Bucket not found"}},
            "HeadBucket",
        )

        result = await s3_service.health_check()

        assert result["status"] == "unhealthy"
        assert "S3 error (NoSuchBucket)" in result["error"]

    @pytest.mark.asyncio
    async def test_health_check_unexpected_error(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test health check with unexpected error."""
        mock_s3_client.head_bucket.side_effect = BotoCoreError()

        result = await s3_service.health_check()

        assert result["status"] == "unhealthy"
        assert "Connection error" in result["error"]


class TestCloudStoragePortMethods:
    """Test CloudStoragePort interface methods."""

    @pytest.mark.asyncio
    async def test_upload_file(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test generic file upload."""
        file_data = b"test file content"
        file_path = "test/path/file.txt"
        metadata = {"content-type": "text/plain"}

        s3_uri = await s3_service.upload_file(file_data, file_path, metadata)

        assert s3_uri == f"s3://test-health-bucket/{file_path}"

        mock_s3_client.put_object.assert_called_once()
        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert call_kwargs["Body"] == file_data
        assert call_kwargs["Key"] == file_path
        assert call_kwargs["Metadata"] == metadata

    @pytest.mark.asyncio
    async def test_upload_file_error(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test file upload with error."""
        mock_s3_client.put_object.side_effect = Exception("Simulated upload error")

        with pytest.raises(S3UploadError) as exc_info:
            await s3_service.upload_file(b"data", "test/file.txt")

        assert "File upload failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_download_file(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test generic file download."""
        file_data = b"test file content"
        mock_response = {"Body": MagicMock()}
        mock_response["Body"].read.return_value = file_data
        mock_s3_client.get_object.return_value = mock_response

        result = await s3_service.download_file("test/path/file.txt")

        assert result == file_data

    @pytest.mark.asyncio
    async def test_download_file_error(
        self, s3_service: S3StorageService, mock_s3_client: MagicMock
    ) -> None:
        """Test file download with error."""
        mock_s3_client.get_object.side_effect = Exception("Simulated download error")

        with pytest.raises(S3DownloadError) as exc_info:
            await s3_service.download_file("test/file.txt")

        assert "File download failed" in str(exc_info.value)

    def test_bucket_method(self, s3_service: S3StorageService) -> None:
        """Test bucket method."""
        result = s3_service.bucket("test-bucket")
        assert result == "test-bucket"

    def test_get_raw_data_bucket_name(self, s3_service: S3StorageService) -> None:
        """Test get_raw_data_bucket_name method."""
        assert s3_service.get_raw_data_bucket_name() == "test-health-bucket"

    def test_upload_json(self, s3_service: S3StorageService) -> None:
        """Test JSON upload method."""
        # This is a synchronous method that needs special handling

        async def mock_upload_file(
            _data: bytes, file_path: str, _metadata: dict[str, Any] | None = None
        ) -> str:
            await asyncio.sleep(0)  # Make it properly async
            return f"s3://test-bucket/{file_path}"

        with patch.object(s3_service, "upload_file", side_effect=mock_upload_file):
            data = {"test": "data", "number": 123}

            # Run the synchronous method that internally uses asyncio.run
            result = s3_service.upload_json(
                "test-bucket", "test.json", data, metadata={"type": "test"}
            )

            assert result == "s3://test-bucket/test.json"


class TestGlobalFunctions:
    """Test global functions."""

    def test_get_s3_service(self) -> None:
        """Test get_s3_service singleton."""
        with (
            patch("clarity.services.s3_storage_service._s3_service", None),
            patch("boto3.client"),
        ):
            service1 = get_s3_service("test-bucket")
            service2 = get_s3_service("test-bucket")

            assert service1 is service2  # Same instance

    def test_get_s3_service_default_bucket(self) -> None:
        """Test get_s3_service with default bucket name."""
        with (
            patch("clarity.services.s3_storage_service._s3_service", None),
            patch("boto3.client"),
        ):
            service = get_s3_service()  # No bucket name provided

            assert service.bucket_name == "clarity-health-data-storage"


class TestExceptionClasses:
    """Test custom exception classes."""

    def test_s3_storage_error(self) -> None:
        """Test S3StorageError exception."""
        error = S3StorageError("Test error")
        assert str(error) == "Test error"

    def test_s3_upload_error(self) -> None:
        """Test S3UploadError exception."""
        error = S3UploadError("Upload failed")
        assert str(error) == "Upload failed"
        assert isinstance(error, S3StorageError)

    def test_s3_download_error(self) -> None:
        """Test S3DownloadError exception."""
        error = S3DownloadError("Download failed")
        assert str(error) == "Download failed"
        assert isinstance(error, S3StorageError)

    def test_s3_permission_error(self) -> None:
        """Test S3PermissionError exception."""
        error = S3PermissionError("Access denied")
        assert str(error) == "Access denied"
        assert isinstance(error, S3StorageError)
