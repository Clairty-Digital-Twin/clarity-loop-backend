"""Comprehensive tests for DynamoDB service."""

from __future__ import annotations

from datetime import UTC, datetime
import time
from typing import Any
from unittest.mock import MagicMock, patch
import uuid

from botocore.exceptions import ClientError
import pytest

from clarity.services.dynamodb_service import (
    DocumentNotFoundError,
    DynamoDBConnectionError,
    DynamoDBError,
    DynamoDBHealthDataRepository,
    DynamoDBPermissionError,
    DynamoDBService,
    DynamoDBValidationError,
)


@pytest.fixture
def mock_dynamodb_resource() -> MagicMock:
    """Mock DynamoDB resource."""
    return MagicMock()


@pytest.fixture
def mock_table() -> MagicMock:
    """Mock DynamoDB table."""
    table = MagicMock()
    table.put_item = MagicMock()
    table.get_item = MagicMock()
    table.delete_item = MagicMock()
    table.update_item = MagicMock()
    table.query = MagicMock()
    table.scan = MagicMock()
    table.batch_write_item = MagicMock()
    return table


@pytest.fixture
def dynamodb_service(
    mock_dynamodb_resource: MagicMock, mock_table: MagicMock
) -> DynamoDBService:
    """Create DynamoDB service with mocked resource."""
    service = DynamoDBService(
        region="us-east-1",
        table_prefix="test_",
        enable_caching=True,
        cache_ttl=300,
    )
    service.dynamodb = mock_dynamodb_resource

    # Create separate mock for audit table
    mock_audit_table = MagicMock()
    mock_audit_table.put_item = MagicMock()

    # Return different mocks based on table name
    def get_table(table_name: str) -> MagicMock:
        if "audit" in table_name:
            return mock_audit_table
        return mock_table

    mock_dynamodb_resource.Table.side_effect = get_table
    return service


@pytest.fixture
def dynamodb_service_no_cache(
    mock_dynamodb_resource: MagicMock, mock_table: MagicMock
) -> DynamoDBService:
    """Create DynamoDB service without caching."""
    service = DynamoDBService(
        region="us-east-1",
        table_prefix="test_",
        enable_caching=False,
    )
    service.dynamodb = mock_dynamodb_resource
    mock_dynamodb_resource.Table.return_value = mock_table
    return service


@pytest.fixture
def valid_health_data() -> dict[str, Any]:
    """Create valid health data."""
    return {
        "user_id": str(uuid.uuid4()),
        "processing_id": str(uuid.uuid4()),
        "upload_source": "mobile_app",
        "metrics": [
            {
                "metric_id": str(uuid.uuid4()),
                "metric_type": "heart_rate",
                "value": 72,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            {
                "metric_id": str(uuid.uuid4()),
                "metric_type": "steps",
                "value": 5000,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        ],
    }


class TestDynamoDBServiceInit:
    """Test DynamoDB service initialization."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        with patch("boto3.resource") as mock_boto_resource:
            service = DynamoDBService()

            assert service.region == "us-east-1"
            assert service.endpoint_url is None
            assert service.table_prefix == "clarity_"
            assert service.enable_caching is True
            assert service.cache_ttl == 300

            mock_boto_resource.assert_called_once_with(
                "dynamodb",
                region_name="us-east-1",
                endpoint_url=None,
            )

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        with patch("boto3.resource") as mock_boto_resource:
            service = DynamoDBService(
                region="eu-west-1",
                endpoint_url="http://localhost:8000",
                table_prefix="custom_",
                enable_caching=False,
                cache_ttl=600,
            )

            assert service.region == "eu-west-1"
            assert service.endpoint_url == "http://localhost:8000"
            assert service.table_prefix == "custom_"
            assert service.enable_caching is False
            assert service.cache_ttl == 600

            mock_boto_resource.assert_called_once_with(
                "dynamodb",
                region_name="eu-west-1",
                endpoint_url="http://localhost:8000",
            )

    def test_table_names(self, dynamodb_service: DynamoDBService) -> None:
        """Test table name generation."""
        assert dynamodb_service.tables["health_data"] == "test_health_data"
        assert dynamodb_service.tables["processing_jobs"] == "test_processing_jobs"
        assert dynamodb_service.tables["user_profiles"] == "test_user_profiles"
        assert dynamodb_service.tables["audit_logs"] == "test_audit_logs"


class TestCachingMethods:
    """Test caching functionality."""

    def test_cache_key_generation(self, dynamodb_service: DynamoDBService) -> None:
        """Test cache key generation."""
        key = dynamodb_service._cache_key("test_table", "item123")
        assert key == "test_table:item123"

    def test_is_cache_valid_enabled(self, dynamodb_service: DynamoDBService) -> None:
        """Test cache validity check when enabled."""
        # Fresh cache entry
        cache_entry = {
            "data": {"test": "data"},
            "timestamp": time.time(),
        }
        assert dynamodb_service._is_cache_valid(cache_entry) is True

        # Expired cache entry
        old_entry = {
            "data": {"test": "data"},
            "timestamp": time.time() - 400,  # Older than TTL
        }
        assert dynamodb_service._is_cache_valid(old_entry) is False

    def test_is_cache_valid_disabled(
        self, dynamodb_service_no_cache: DynamoDBService
    ) -> None:
        """Test cache validity check when disabled."""
        cache_entry = {
            "data": {"test": "data"},
            "timestamp": time.time(),
        }
        assert dynamodb_service_no_cache._is_cache_valid(cache_entry) is False


class TestValidateHealthData:
    """Test health data validation."""

    @pytest.mark.asyncio
    async def test_validate_health_data_success(
        self, dynamodb_service: DynamoDBService, valid_health_data: dict[str, Any]
    ) -> None:
        """Test successful health data validation."""
        # Should not raise
        await dynamodb_service._validate_health_data(valid_health_data)

    @pytest.mark.asyncio
    async def test_validate_health_data_missing_field(
        self, dynamodb_service: DynamoDBService
    ) -> None:
        """Test validation with missing required field."""
        invalid_data = {
            "user_id": str(uuid.uuid4()),
            # Missing "metrics" and "upload_source"
        }

        with pytest.raises(
            DynamoDBValidationError, match="Missing required field: metrics"
        ):
            await dynamodb_service._validate_health_data(invalid_data)

    @pytest.mark.asyncio
    async def test_validate_health_data_invalid_user_id(
        self, dynamodb_service: DynamoDBService
    ) -> None:
        """Test validation with invalid user_id format."""
        invalid_data = {
            "user_id": "not-a-uuid",
            "metrics": [{"test": "data"}],
            "upload_source": "test",
        }

        with pytest.raises(DynamoDBValidationError, match="Invalid user_id format"):
            await dynamodb_service._validate_health_data(invalid_data)

    @pytest.mark.asyncio
    async def test_validate_health_data_empty_metrics(
        self, dynamodb_service: DynamoDBService
    ) -> None:
        """Test validation with empty metrics list."""
        invalid_data = {
            "user_id": str(uuid.uuid4()),
            "metrics": [],
            "upload_source": "test",
        }

        with pytest.raises(
            DynamoDBValidationError, match="Metrics must be a non-empty list"
        ):
            await dynamodb_service._validate_health_data(invalid_data)

    @pytest.mark.asyncio
    async def test_validate_health_data_invalid_metrics_type(
        self, dynamodb_service: DynamoDBService
    ) -> None:
        """Test validation with invalid metrics type."""
        invalid_data = {
            "user_id": str(uuid.uuid4()),
            "metrics": "not-a-list",
            "upload_source": "test",
        }

        with pytest.raises(
            DynamoDBValidationError, match="Metrics must be a non-empty list"
        ):
            await dynamodb_service._validate_health_data(invalid_data)


class TestAuditLog:
    """Test audit logging functionality."""

    @pytest.mark.asyncio
    async def test_audit_log_success(
        self, dynamodb_service: DynamoDBService, mock_dynamodb_resource: MagicMock
    ) -> None:
        """Test successful audit log creation."""
        # Get the mock audit table that was created by the fixture
        mock_audit_table = mock_dynamodb_resource.Table("test_audit_logs")

        await dynamodb_service._audit_log(
            operation="test_operation",
            table="test_table",
            item_id="item123",
            user_id="user456",
            metadata={"action": "test"},
        )

        # Verify table was called
        mock_dynamodb_resource.Table.assert_called_with("test_audit_logs")
        mock_audit_table.put_item.assert_called_once()

        # Check audit entry
        call_args = mock_audit_table.put_item.call_args[1]
        audit_entry = call_args["Item"]
        assert audit_entry["operation"] == "test_operation"
        assert audit_entry["table"] == "test_table"
        assert audit_entry["item_id"] == "item123"
        assert audit_entry["user_id"] == "user456"
        assert audit_entry["metadata"] == {"action": "test"}
        assert "audit_id" in audit_entry
        assert "timestamp" in audit_entry

    @pytest.mark.asyncio
    async def test_audit_log_failure(
        self, dynamodb_service: DynamoDBService, mock_dynamodb_resource: MagicMock
    ) -> None:
        """Test audit log creation failure (should not raise)."""
        # Get the mock audit table that was created by the fixture
        mock_audit_table = mock_dynamodb_resource.Table("test_audit_logs")
        mock_audit_table.put_item.side_effect = Exception("Audit error")

        # Should not raise exception
        await dynamodb_service._audit_log(
            operation="test_operation",
            table="test_table",
            item_id="item123",
        )


class TestPutItem:
    """Test put_item functionality."""

    @pytest.mark.asyncio
    async def test_put_item_success(
        self,
        dynamodb_service: DynamoDBService,
        mock_table: MagicMock,
        valid_health_data: dict[str, Any],
    ) -> None:
        """Test successful item creation."""
        item_id = await dynamodb_service.put_item(
            table_name="test_health_data",  # Use full table name
            item=valid_health_data.copy(),
            user_id="user123",
        )

        # The service uses user_id as ID when available
        assert item_id == valid_health_data["user_id"]
        mock_table.put_item.assert_called_once()

        # Check item modifications
        call_args = mock_table.put_item.call_args[1]
        stored_item = call_args["Item"]
        assert "created_at" in stored_item
        assert "updated_at" in stored_item
        assert stored_item["id"] == valid_health_data["user_id"]

    @pytest.mark.asyncio
    async def test_put_item_generate_id(
        self, dynamodb_service: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test put_item with generated ID."""
        item_data = {
            "metrics": [{"test": "data"}],
            "upload_source": "test",
        }

        # Don't include user_id so it generates a new ID
        test_uuid = uuid.uuid4()
        with patch("uuid.uuid4", return_value=test_uuid):
            item_id = await dynamodb_service.put_item(
                table_name="test_other_table",  # Not health_data to skip validation
                item=item_data.copy(),
            )

        assert item_id == str(test_uuid)
        call_args = mock_table.put_item.call_args[1]
        assert call_args["Item"]["id"] == str(test_uuid)

    @pytest.mark.asyncio
    async def test_put_item_validation_failure(
        self, dynamodb_service: DynamoDBService
    ) -> None:
        """Test put_item with validation failure."""
        invalid_data = {
            "user_id": "not-a-uuid",  # Invalid UUID format
            "metrics": [{"test": "data"}],
            "upload_source": "test",
        }

        with pytest.raises(DynamoDBValidationError):
            await dynamodb_service.put_item(
                table_name="test_health_data",  # Use full table name
                item=invalid_data,
            )

    @pytest.mark.asyncio
    async def test_put_item_client_error(
        self,
        dynamodb_service: DynamoDBService,
        mock_table: MagicMock,
        valid_health_data: dict[str, Any],
    ) -> None:
        """Test put_item with ClientError."""
        mock_table.put_item.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException"}}, "PutItem"
        )

        with pytest.raises(DynamoDBError, match="ResourceNotFoundException"):
            await dynamodb_service.put_item(
                table_name="test_health_data",  # Use full table name
                item=valid_health_data.copy(),
            )


class TestGetItem:
    """Test get_item functionality."""

    @pytest.mark.asyncio
    async def test_get_item_success_cached(
        self, dynamodb_service: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test successful item retrieval from cache."""
        # Pre-populate cache
        cache_key = dynamodb_service._cache_key("test_table", "item123")
        dynamodb_service._cache[cache_key] = {
            "data": {"id": "item123", "test": "cached_data"},
            "timestamp": time.time(),
        }

        # Pass key as dict as expected by the method
        result = await dynamodb_service.get_item("test_table", {"id": "item123"})

        assert result == {"id": "item123", "test": "cached_data"}
        # Should not call DynamoDB
        mock_table.get_item.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_item_success_from_db(
        self, dynamodb_service: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test successful item retrieval from database."""
        mock_table.get_item.return_value = {
            "Item": {"id": "item123", "test": "db_data"}
        }

        result = await dynamodb_service.get_item("test_table", {"id": "item123"})

        assert result == {"id": "item123", "test": "db_data"}
        mock_table.get_item.assert_called_once_with(Key={"id": "item123"})

        # Check cache was updated
        cache_key = dynamodb_service._cache_key("test_table", "item123")
        assert cache_key in dynamodb_service._cache
        assert dynamodb_service._cache[cache_key]["data"] == result

    @pytest.mark.asyncio
    async def test_get_item_not_found(
        self, dynamodb_service: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test get_item when item doesn't exist."""
        mock_table.get_item.return_value = {}  # No Item key

        result = await dynamodb_service.get_item("test_table", {"id": "missing123"})

        assert result is None

    @pytest.mark.asyncio
    async def test_get_item_no_cache(
        self, dynamodb_service_no_cache: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test get_item without caching."""
        mock_table.get_item.return_value = {"Item": {"id": "item123", "test": "data"}}

        result = await dynamodb_service_no_cache.get_item(
            "test_table", {"id": "item123"}
        )

        assert result == {"id": "item123", "test": "data"}
        # Cache should be empty
        assert len(dynamodb_service_no_cache._cache) == 0


class TestDeleteItem:
    """Test delete_item functionality."""

    @pytest.mark.asyncio
    async def test_delete_item_success(
        self, dynamodb_service: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test successful item deletion."""
        # Pre-populate cache
        cache_key = dynamodb_service._cache_key("test_table", "item123")
        dynamodb_service._cache[cache_key] = {
            "data": {"test": "data"},
            "timestamp": time.time(),
        }

        result = await dynamodb_service.delete_item(
            "test_table", {"id": "item123"}, user_id="user456"
        )

        assert result is True
        mock_table.delete_item.assert_called_once_with(Key={"id": "item123"})
        # Cache should be cleared
        assert cache_key not in dynamodb_service._cache

    @pytest.mark.asyncio
    async def test_delete_item_not_found(
        self, dynamodb_service: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test delete_item when item doesn't exist."""
        # Delete should succeed even if item doesn't exist
        result = await dynamodb_service.delete_item("test_table", {"id": "missing123"})

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_item_other_error(
        self, dynamodb_service: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test delete_item with other errors."""
        mock_table.delete_item.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}}, "DeleteItem"
        )

        with pytest.raises(DynamoDBError):
            await dynamodb_service.delete_item("test_table", {"id": "item123"})


class TestQueryItems:
    """Test query_items functionality."""

    @pytest.mark.asyncio
    async def test_query_items_success(
        self, dynamodb_service: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test successful query operation."""
        mock_table.query.return_value = {
            "Items": [
                {"id": "1", "user_id": "user123", "data": "test1"},
                {"id": "2", "user_id": "user123", "data": "test2"},
            ],
            "Count": 2,
        }

        response = await dynamodb_service.query(
            table_name="health_data",
            key_condition_expression="user_id = :uid",
            expression_attribute_values={":uid": "user123"},
            limit=10,
        )

        results = response["Items"]
        assert len(results) == 2
        assert results[0]["id"] == "1"
        assert results[1]["id"] == "2"

        mock_table.query.assert_called_once()
        call_args = mock_table.query.call_args[1]
        assert call_args["KeyConditionExpression"] == "user_id = :uid"
        assert call_args["ExpressionAttributeValues"] == {":uid": "user123"}
        assert call_args["Limit"] == 10

    @pytest.mark.asyncio
    async def test_query_items_with_filter(
        self, dynamodb_service: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test query with filter expression."""
        mock_table.query.return_value = {"Items": [], "Count": 0}

        await dynamodb_service.query(
            table_name="health_data",
            key_condition_expression="user_id = :uid",
            expression_attribute_values={":uid": "user123"},
            limit=None,
            scan_index_forward=False,
        )

        call_args = mock_table.query.call_args[1]
        assert call_args["ScanIndexForward"] is False

    @pytest.mark.asyncio
    async def test_query_items_empty_result(
        self, dynamodb_service: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test query with no results."""
        mock_table.query.return_value = {"Items": [], "Count": 0}

        response = await dynamodb_service.query(
            table_name="test_table",
            key_condition_expression="id = :id",
            expression_attribute_values={":id": "missing"},
        )

        assert response["Items"] == []


class TestUpdateItem:
    """Test update_item functionality."""

    @pytest.mark.asyncio
    async def test_update_item_success(
        self, dynamodb_service: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test successful item update."""
        mock_table.update_item.return_value = {
            "Attributes": {
                "id": "item123",
                "status": "updated",
                "updated_at": datetime.now(UTC).isoformat(),
            }
        }

        # Clear cache to ensure update
        cache_key = dynamodb_service._cache_key("test_table", "item123")
        dynamodb_service._cache[cache_key] = {
            "data": {"old": "data"},
            "timestamp": time.time(),
        }

        result = await dynamodb_service.update_item(
            table_name="test_table",
            key={"id": "item123"},
            update_expression="SET #status = :status",
            expression_attribute_values={":status": "updated"},
            user_id="user456",
        )

        assert result is True
        # Cache should be cleared
        assert cache_key not in dynamodb_service._cache

    @pytest.mark.asyncio
    async def test_update_item_not_found(
        self, dynamodb_service: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test update_item when item doesn't exist."""
        mock_table.update_item.side_effect = ClientError(
            {"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem"
        )

        result = await dynamodb_service.update_item(
            table_name="test_table",
            key={"id": "missing123"},
            update_expression="SET #status = :status",
            expression_attribute_values={":status": "updated"},
        )

        assert result is False


class TestBatchOperations:
    """Test batch operations."""

    @pytest.mark.asyncio
    async def test_batch_write_items_success(
        self, dynamodb_service: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test successful batch write."""
        items = [
            {"id": "1", "data": "test1"},
            {"id": "2", "data": "test2"},
            {"id": "3", "data": "test3"},
        ]

        await dynamodb_service.batch_write_items("test_table", items)
        # Should be called with batch writer context
        assert mock_table.batch_writer.called

    @pytest.mark.asyncio
    async def test_batch_write_items_empty_list(
        self, dynamodb_service: DynamoDBService
    ) -> None:
        """Test batch write with empty list."""
        # Should not raise any exception
        await dynamodb_service.batch_write_items("test_table", [])
        # Should not attempt any writes

    @pytest.mark.asyncio
    async def test_batch_write_items_error(
        self, dynamodb_service: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test batch write with error."""
        mock_batch_writer = MagicMock()
        mock_batch_writer.__enter__.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}}, "BatchWriteItem"
        )
        mock_table.batch_writer.return_value = mock_batch_writer

        with pytest.raises(DynamoDBError):
            await dynamodb_service.batch_write_items("test_table", [{"id": "1"}])


class TestHealthDataRepository:
    """Test IHealthDataRepository implementation."""

    @pytest.mark.asyncio
    async def test_save_health_data_success(
        self, dynamodb_service: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test saving health data."""
        user_id = str(uuid.uuid4())
        processing_id = str(uuid.uuid4())
        metrics = [
            {"metric_id": "1", "type": "heart_rate", "value": 72},
            {"metric_id": "2", "type": "steps", "value": 5000},
        ]

        # Create a health data repository instead
        repository = DynamoDBHealthDataRepository()
        repository._dynamodb_service = dynamodb_service

        result = await repository.save_health_data(
            user_id=user_id,
            processing_id=processing_id,
            metrics=metrics,
            upload_source="mobile_app",
            client_timestamp=datetime.now(UTC),
        )

        assert result is True
        # Should have called put_item for processing job and batch_write for metrics
        assert mock_table.put_item.called

    @pytest.mark.asyncio
    async def test_get_processing_status_success(
        self, dynamodb_service: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test getting processing status."""
        mock_table.get_item.return_value = {
            "Item": {
                "processing_id": "proc123",
                "user_id": "user123",
                "status": "completed",
                "total_metrics": 10,
                "processed_metrics": 10,
            }
        }

        # Create a health data repository instead
        repository = DynamoDBHealthDataRepository()
        repository._dynamodb_service = dynamodb_service

        result = await repository.get_processing_status("proc123", "user123")

        assert result["status"] == "completed"
        assert result["total_metrics"] == 10

    @pytest.mark.asyncio
    async def test_get_user_health_data_success(
        self, dynamodb_service: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test getting user health data."""
        mock_table.query.return_value = {
            "Items": [
                {"metric_id": "1", "type": "heart_rate", "value": 72},
                {"metric_id": "2", "type": "steps", "value": 5000},
            ],
            "Count": 2,
        }

        # Create a health data repository instead
        repository = DynamoDBHealthDataRepository()
        repository._dynamodb_service = dynamodb_service

        result = await repository.get_user_health_data(
            user_id="user123",
            limit=10,
            offset=0,
        )

        assert len(result["metrics"]) == 2
        assert result["pagination"]["total"] == 2
        assert result["pagination"]["limit"] == 10
        assert result["pagination"]["offset"] == 0

    @pytest.mark.asyncio
    async def test_delete_health_data_success(
        self, dynamodb_service: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test deleting health data."""
        # Mock query to return items to delete
        mock_table.query.return_value = {
            "Items": [
                {"id": "1", "processing_id": "proc123"},
                {"id": "2", "processing_id": "proc123"},
            ]
        }

        # Create a health data repository instead
        repository = DynamoDBHealthDataRepository()
        repository._dynamodb_service = dynamodb_service

        result = await repository.delete_health_data("user123", "proc123")

        assert result is True
        # Should query for items then delete each
        mock_table.query.assert_called()


# Removed TestScanOperation as scan_table method doesn't exist


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_success(
        self, dynamodb_service: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test successful health check."""
        mock_table.load = MagicMock()

        result = await dynamodb_service.health_check()

        assert result["status"] == "healthy"
        assert result["region"] == "us-east-1"
        assert "cache_enabled" in result
        assert "cached_items" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_health_check_failure(
        self, dynamodb_service: DynamoDBService, mock_table: MagicMock
    ) -> None:
        """Test health check with failure."""
        mock_table.load = MagicMock(side_effect=Exception("Connection error"))

        result = await dynamodb_service.health_check()

        assert result["status"] == "unhealthy"
        assert "error" in result
        assert "Connection error" in result["error"]


# Removed TestTableManagement as create_tables and delete_tables methods don't exist


class TestExceptionClasses:
    """Test custom exception classes."""

    def test_dynamodb_error(self) -> None:
        """Test DynamoDBError exception."""
        error = DynamoDBError("Test error")
        assert str(error) == "Test error"

    def test_document_not_found_error(self) -> None:
        """Test DocumentNotFoundError exception."""
        error = DocumentNotFoundError("Document not found")
        assert str(error) == "Document not found"
        assert isinstance(error, DynamoDBError)

    def test_dynamodb_permission_error(self) -> None:
        """Test DynamoDBPermissionError exception."""
        error = DynamoDBPermissionError("Permission denied")
        assert str(error) == "Permission denied"
        assert isinstance(error, DynamoDBError)

    def test_dynamodb_validation_error(self) -> None:
        """Test DynamoDBValidationError exception."""
        error = DynamoDBValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert isinstance(error, DynamoDBError)

    def test_dynamodb_connection_error(self) -> None:
        """Test DynamoDBConnectionError exception."""
        error = DynamoDBConnectionError("Connection failed")
        assert str(error) == "Connection failed"
        assert isinstance(error, DynamoDBError)
