"""Comprehensive tests for DynamoDB service."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import uuid

from botocore.exceptions import ClientError
import pytest

from clarity.services.dynamodb_service import (
    DocumentNotFoundError,
    DynamoDBConnectionError,
    DynamoDBError,
    DynamoDBPermissionError,
    DynamoDBService,
    DynamoDBValidationError,
)


@pytest.fixture
def mock_dynamodb_resource():
    """Mock DynamoDB resource."""
    with patch("boto3.resource") as mock_resource:
        yield mock_resource


@pytest.fixture
def dynamodb_service(mock_dynamodb_resource):
    """Create DynamoDB service with mocked resource."""
    service = DynamoDBService(
        region="us-east-1",
        table_prefix="test_",
        enable_caching=True,
        cache_ttl=300,
    )
    return service


@pytest.fixture
def mock_table():
    """Create a mock DynamoDB table."""
    table = MagicMock()
    table.put_item = MagicMock()
    table.get_item = MagicMock()
    table.update_item = MagicMock()
    table.delete_item = MagicMock()
    table.query = MagicMock()
    table.load = MagicMock()
    table.batch_writer = MagicMock()
    return table


@pytest.fixture
def valid_health_data():
    """Create valid health data."""
    return {
        "user_id": str(uuid.uuid4()),
        "metrics": [
            {"type": "heart_rate", "value": 72, "timestamp": "2024-01-01T12:00:00Z"},
            {"type": "steps", "value": 5000, "timestamp": "2024-01-01T12:00:00Z"},
        ],
        "upload_source": "mobile_app",
    }


class TestDynamoDBServiceInit:
    """Test DynamoDB service initialization."""

    def test_init_default_params(self, mock_dynamodb_resource):
        """Test initialization with default parameters."""
        service = DynamoDBService()

        assert service.region == "us-east-1"
        assert service.endpoint_url is None
        assert service.table_prefix == "clarity_"
        assert service.enable_caching is True
        assert service.cache_ttl == 300

        mock_dynamodb_resource.assert_called_once_with(
            "dynamodb",
            region_name="us-east-1",
            endpoint_url=None,
        )

    def test_init_custom_params(self, mock_dynamodb_resource):
        """Test initialization with custom parameters."""
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

        mock_dynamodb_resource.assert_called_once_with(
            "dynamodb",
            region_name="eu-west-1",
            endpoint_url="http://localhost:8000",
        )

    def test_table_names(self, dynamodb_service):
        """Test table name configuration."""
        expected_tables = {
            "health_data": "test_health_data",
            "processing_jobs": "test_processing_jobs",
            "user_profiles": "test_user_profiles",
            "audit_logs": "test_audit_logs",
            "ml_models": "test_ml_models",
            "insights": "test_insights",
            "analysis_results": "test_analysis_results",
        }

        assert dynamodb_service.tables == expected_tables


class TestCachingMethods:
    """Test caching-related methods."""

    def test_cache_key_generation(self, dynamodb_service):
        """Test cache key generation."""
        key = dynamodb_service._cache_key("test_table", "test_id")
        assert key == "test_table:test_id"

    def test_is_cache_valid_enabled(self, dynamodb_service):
        """Test cache validity check when caching is enabled."""
        # Fresh cache entry
        cache_entry = {"timestamp": time.time(), "data": {"test": "data"}}
        assert dynamodb_service._is_cache_valid(cache_entry) is True

        # Expired cache entry
        old_timestamp = time.time() - 400  # 400 seconds ago
        expired_entry = {"timestamp": old_timestamp, "data": {"test": "data"}}
        assert dynamodb_service._is_cache_valid(expired_entry) is False

    def test_is_cache_valid_disabled(self, dynamodb_service):
        """Test cache validity check when caching is disabled."""
        dynamodb_service.enable_caching = False

        cache_entry = {"timestamp": time.time(), "data": {"test": "data"}}
        assert dynamodb_service._is_cache_valid(cache_entry) is False


class TestValidateHealthData:
    """Test health data validation."""

    @pytest.mark.asyncio
    async def test_validate_health_data_valid(
        self, dynamodb_service, valid_health_data
    ):
        """Test validation with valid health data."""
        # Should not raise any exception
        await dynamodb_service._validate_health_data(valid_health_data)

    @pytest.mark.asyncio
    async def test_validate_health_data_missing_field(self, dynamodb_service):
        """Test validation with missing required field."""
        invalid_data = {
            "user_id": str(uuid.uuid4()),
            # Missing metrics and upload_source
        }

        with pytest.raises(DynamoDBValidationError) as exc_info:
            await dynamodb_service._validate_health_data(invalid_data)

        assert "Missing required field" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_validate_health_data_invalid_user_id(self, dynamodb_service):
        """Test validation with invalid user_id format."""
        invalid_data = {
            "user_id": "not-a-uuid",
            "metrics": [{"type": "test"}],
            "upload_source": "test",
        }

        with pytest.raises(DynamoDBValidationError) as exc_info:
            await dynamodb_service._validate_health_data(invalid_data)

        assert "Invalid user_id format" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_validate_health_data_empty_metrics(self, dynamodb_service):
        """Test validation with empty metrics list."""
        invalid_data = {
            "user_id": str(uuid.uuid4()),
            "metrics": [],
            "upload_source": "test",
        }

        with pytest.raises(DynamoDBValidationError) as exc_info:
            await dynamodb_service._validate_health_data(invalid_data)

        assert "Metrics must be a non-empty list" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_validate_health_data_invalid_metrics_type(self, dynamodb_service):
        """Test validation with invalid metrics type."""
        invalid_data = {
            "user_id": str(uuid.uuid4()),
            "metrics": "not-a-list",
            "upload_source": "test",
        }

        with pytest.raises(DynamoDBValidationError) as exc_info:
            await dynamodb_service._validate_health_data(invalid_data)

        assert "Metrics must be a non-empty list" in str(exc_info.value)


class TestAuditLog:
    """Test audit logging functionality."""

    @pytest.mark.asyncio
    async def test_audit_log_success(self, dynamodb_service, mock_table):
        """Test successful audit log creation."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)

        with patch("uuid.uuid4", return_value="test-audit-id"):
            await dynamodb_service._audit_log(
                operation="CREATE",
                table="test_table",
                item_id="test_id",
                user_id="test_user",
                metadata={"size": 100},
            )

        # Verify put_item was called
        mock_table.put_item.assert_called_once()
        call_args = mock_table.put_item.call_args[1]["Item"]

        assert call_args["audit_id"] == "test-audit-id"
        assert call_args["operation"] == "CREATE"
        assert call_args["table"] == "test_table"
        assert call_args["item_id"] == "test_id"
        assert call_args["user_id"] == "test_user"
        assert call_args["metadata"] == {"size": 100}
        assert call_args["source"] == "dynamodb_service"
        assert "timestamp" in call_args

    @pytest.mark.asyncio
    async def test_audit_log_failure(self, dynamodb_service, mock_table):
        """Test audit log creation failure (should not raise)."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)
        mock_table.put_item.side_effect = Exception("Audit error")

        # Should not raise exception
        await dynamodb_service._audit_log(
            operation="CREATE",
            table="test_table",
            item_id="test_id",
        )


class TestPutItem:
    """Test put_item functionality."""

    @pytest.mark.asyncio
    async def test_put_item_success(self, dynamodb_service, mock_table):
        """Test successful item creation."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)

        item = {"name": "Test Item", "value": 123}

        with patch("uuid.uuid4", return_value="generated-id"):
            item_id = await dynamodb_service.put_item(
                "test_table", item, user_id="test_user"
            )

        assert item_id == "generated-id"

        # Verify put_item was called twice (once for item, once for audit log)
        assert mock_table.put_item.call_count == 2

        # Check the first call (actual item)
        saved_item = mock_table.put_item.call_args_list[0][1]["Item"]

        assert saved_item["id"] == "generated-id"
        assert saved_item["name"] == "Test Item"
        assert saved_item["value"] == 123
        assert "created_at" in saved_item
        assert "updated_at" in saved_item

        # Verify item is cached
        cache_key = dynamodb_service._cache_key("test_table", "generated-id")
        assert cache_key in dynamodb_service._cache

    @pytest.mark.asyncio
    async def test_put_item_with_existing_id(self, dynamodb_service, mock_table):
        """Test item creation with existing ID."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)

        item = {"id": "existing-id", "name": "Test Item"}

        item_id = await dynamodb_service.put_item("test_table", item)

        assert item_id == "existing-id"

    @pytest.mark.asyncio
    async def test_put_item_health_data_validation(
        self, dynamodb_service, mock_table, valid_health_data
    ):
        """Test health data validation during put_item."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)

        item_id = await dynamodb_service.put_item(
            dynamodb_service.tables["health_data"],
            valid_health_data,
        )

        assert item_id == valid_health_data["user_id"]

    @pytest.mark.asyncio
    async def test_put_item_validation_error(self, dynamodb_service, mock_table):
        """Test put_item with validation error."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)

        invalid_health_data = {"user_id": "invalid"}

        with pytest.raises(DynamoDBValidationError):
            await dynamodb_service.put_item(
                dynamodb_service.tables["health_data"],
                invalid_health_data,
            )

    @pytest.mark.asyncio
    async def test_put_item_client_error(self, dynamodb_service, mock_table):
        """Test put_item with ClientError."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)
        mock_table.put_item.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException"}},
            "PutItem",
        )

        with pytest.raises(DynamoDBError) as exc_info:
            await dynamodb_service.put_item("test_table", {"name": "test"})

        assert "Item creation failed" in str(exc_info.value)


class TestGetItem:
    """Test get_item functionality."""

    @pytest.mark.asyncio
    async def test_get_item_success(self, dynamodb_service, mock_table):
        """Test successful item retrieval."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)

        test_item = {"id": "test-id", "name": "Test Item", "value": 123}
        mock_table.get_item.return_value = {"Item": test_item}

        result = await dynamodb_service.get_item("test_table", {"id": "test-id"})

        assert result == test_item
        mock_table.get_item.assert_called_once_with(Key={"id": "test-id"})

        # Verify item is cached
        cache_key = dynamodb_service._cache_key("test_table", "test-id")
        assert cache_key in dynamodb_service._cache

    @pytest.mark.asyncio
    async def test_get_item_from_cache(self, dynamodb_service, mock_table):
        """Test item retrieval from cache."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)

        # Pre-populate cache
        cache_key = dynamodb_service._cache_key("test_table", "test-id")
        cached_item = {"id": "test-id", "name": "Cached Item"}
        dynamodb_service._cache[cache_key] = {
            "data": cached_item,
            "timestamp": time.time(),
        }

        result = await dynamodb_service.get_item("test_table", {"id": "test-id"})

        assert result == cached_item
        # Should not call DynamoDB
        mock_table.get_item.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_item_not_found(self, dynamodb_service, mock_table):
        """Test get_item when item doesn't exist."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)
        mock_table.get_item.return_value = {}  # No 'Item' key

        result = await dynamodb_service.get_item("test_table", {"id": "missing-id"})

        assert result is None

    @pytest.mark.asyncio
    async def test_get_item_error(self, dynamodb_service, mock_table):
        """Test get_item with error."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)
        mock_table.get_item.side_effect = Exception("Get error")

        with pytest.raises(DynamoDBError) as exc_info:
            await dynamodb_service.get_item("test_table", {"id": "test-id"})

        assert "Item retrieval failed" in str(exc_info.value)


class TestUpdateItem:
    """Test update_item functionality."""

    @pytest.mark.asyncio
    async def test_update_item_success(self, dynamodb_service, mock_table):
        """Test successful item update."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)

        result = await dynamodb_service.update_item(
            "test_table",
            {"id": "test-id"},
            "SET #name = :name",
            {":name": "Updated Name"},
            user_id="test_user",
        )

        assert result is True

        # Verify update_item was called
        mock_table.update_item.assert_called_once()
        call_kwargs = mock_table.update_item.call_args[1]

        assert call_kwargs["Key"] == {"id": "test-id"}
        assert "updated_at" in call_kwargs["UpdateExpression"]
        assert ":updated_at" in call_kwargs["ExpressionAttributeValues"]

        # Verify cache is cleared
        cache_key = dynamodb_service._cache_key("test_table", "test-id")
        assert cache_key not in dynamodb_service._cache

    @pytest.mark.asyncio
    async def test_update_item_not_found(self, dynamodb_service, mock_table):
        """Test update_item when item doesn't exist."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)
        mock_table.update_item.side_effect = ClientError(
            {"Error": {"Code": "ConditionalCheckFailedException"}},
            "UpdateItem",
        )

        result = await dynamodb_service.update_item(
            "test_table",
            {"id": "missing-id"},
            "SET #name = :name",
            {":name": "Updated"},
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_update_item_error(self, dynamodb_service, mock_table):
        """Test update_item with error."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)
        mock_table.update_item.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}},
            "UpdateItem",
        )

        with pytest.raises(DynamoDBError) as exc_info:
            await dynamodb_service.update_item(
                "test_table",
                {"id": "test-id"},
                "SET #name = :name",
                {":name": "Updated"},
            )

        assert "Item update failed" in str(exc_info.value)


class TestDeleteItem:
    """Test delete_item functionality."""

    @pytest.mark.asyncio
    async def test_delete_item_success(self, dynamodb_service, mock_table):
        """Test successful item deletion."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)

        # Pre-populate cache
        cache_key = dynamodb_service._cache_key("test_table", "test-id")
        dynamodb_service._cache[cache_key] = {"data": {}, "timestamp": time.time()}

        result = await dynamodb_service.delete_item(
            "test_table",
            {"id": "test-id"},
            user_id="test_user",
        )

        assert result is True
        mock_table.delete_item.assert_called_once_with(Key={"id": "test-id"})

        # Verify cache is cleared
        assert cache_key not in dynamodb_service._cache

    @pytest.mark.asyncio
    async def test_delete_item_error(self, dynamodb_service, mock_table):
        """Test delete_item with error."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)
        mock_table.delete_item.side_effect = Exception("Delete error")

        with pytest.raises(DynamoDBError) as exc_info:
            await dynamodb_service.delete_item("test_table", {"id": "test-id"})

        assert "Item deletion failed" in str(exc_info.value)


class TestQuery:
    """Test query functionality."""

    @pytest.mark.asyncio
    async def test_query_success(self, dynamodb_service, mock_table):
        """Test successful query operation."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)

        mock_response = {
            "Items": [{"id": "1", "name": "Item 1"}, {"id": "2", "name": "Item 2"}],
            "Count": 2,
            "LastEvaluatedKey": {"id": "2"},
        }
        mock_table.query.return_value = mock_response

        result = await dynamodb_service.query(
            "test_table",
            "user_id = :user_id",
            {":user_id": "test-user"},
            limit=10,
            scan_index_forward=False,
        )

        assert result["Count"] == 2
        assert len(result["Items"]) == 2
        assert result["LastEvaluatedKey"] == {"id": "2"}

        # Verify query was called correctly
        mock_table.query.assert_called_once()
        call_kwargs = mock_table.query.call_args[1]
        assert call_kwargs["KeyConditionExpression"] == "user_id = :user_id"
        assert call_kwargs["ExpressionAttributeValues"] == {":user_id": "test-user"}
        assert call_kwargs["ScanIndexForward"] is False
        assert call_kwargs["Limit"] == 10

    @pytest.mark.asyncio
    async def test_query_error(self, dynamodb_service, mock_table):
        """Test query with error."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)
        mock_table.query.side_effect = Exception("Query error")

        with pytest.raises(DynamoDBError) as exc_info:
            await dynamodb_service.query(
                "test_table",
                "user_id = :user_id",
                {":user_id": "test-user"},
            )

        assert "Query operation failed" in str(exc_info.value)


class TestBatchWriteItems:
    """Test batch write functionality."""

    @pytest.mark.asyncio
    async def test_batch_write_items_success(self, dynamodb_service, mock_table):
        """Test successful batch write."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)

        # Mock batch writer
        mock_batch_writer = MagicMock()
        mock_batch_writer.__enter__ = MagicMock(return_value=mock_batch_writer)
        mock_batch_writer.__exit__ = MagicMock(return_value=False)
        mock_table.batch_writer.return_value = mock_batch_writer

        items = [
            {"name": "Item 1"},
            {"name": "Item 2"},
            {"name": "Item 3"},
        ]

        await dynamodb_service.batch_write_items("test_table", items)

        # Verify batch writer was used
        mock_table.batch_writer.assert_called_once()

        # Verify put_item was called for each item
        assert mock_batch_writer.put_item.call_count == 3

        # Verify items have timestamps and IDs
        for call in mock_batch_writer.put_item.call_args_list:
            item = call[1]["Item"]
            assert "id" in item
            assert "created_at" in item
            assert "updated_at" in item

    @pytest.mark.asyncio
    async def test_batch_write_items_large_batch(self, dynamodb_service, mock_table):
        """Test batch write with more than 25 items."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)

        # Mock batch writer
        mock_batch_writer = MagicMock()
        mock_batch_writer.__enter__ = MagicMock(return_value=mock_batch_writer)
        mock_batch_writer.__exit__ = MagicMock(return_value=False)
        mock_table.batch_writer.return_value = mock_batch_writer

        # Create 30 items (more than batch size of 25)
        items = [{"name": f"Item {i}"} for i in range(30)]

        await dynamodb_service.batch_write_items("test_table", items)

        # Should create 2 batches (25 + 5)
        assert mock_table.batch_writer.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_write_items_error(self, dynamodb_service, mock_table):
        """Test batch write with error."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)
        mock_table.batch_writer.side_effect = Exception("Batch write error")

        with pytest.raises(DynamoDBError) as exc_info:
            await dynamodb_service.batch_write_items("test_table", [{"name": "Item"}])

        assert "Batch write operation failed" in str(exc_info.value)


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, dynamodb_service, mock_table):
        """Test successful health check."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)

        result = await dynamodb_service.health_check()

        assert result["status"] == "healthy"
        assert result["region"] == "us-east-1"
        assert result["cache_enabled"] is True
        assert "cached_items" in result
        assert "timestamp" in result

        # Verify table.load was called
        mock_table.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_error(self, dynamodb_service, mock_table):
        """Test health check with error."""
        dynamodb_service.dynamodb.Table = MagicMock(return_value=mock_table)
        mock_table.load.side_effect = Exception("Connection error")

        result = await dynamodb_service.health_check()

        assert result["status"] == "unhealthy"
        assert "error" in result


class TestExceptionClasses:
    """Test custom exception classes."""

    def test_dynamodb_error(self):
        """Test DynamoDBError exception."""
        error = DynamoDBError("Test error")
        assert str(error) == "Test error"

    def test_document_not_found_error(self):
        """Test DocumentNotFoundError exception."""
        error = DocumentNotFoundError("Document not found")
        assert str(error) == "Document not found"
        assert isinstance(error, DynamoDBError)

    def test_dynamodb_permission_error(self):
        """Test DynamoDBPermissionError exception."""
        error = DynamoDBPermissionError("Permission denied")
        assert str(error) == "Permission denied"
        assert isinstance(error, DynamoDBError)

    def test_dynamodb_validation_error(self):
        """Test DynamoDBValidationError exception."""
        error = DynamoDBValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert isinstance(error, DynamoDBError)

    def test_dynamodb_connection_error(self):
        """Test DynamoDBConnectionError exception."""
        error = DynamoDBConnectionError("Connection failed")
        assert str(error) == "Connection failed"
        assert isinstance(error, DynamoDBError)
