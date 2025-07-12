"""Comprehensive tests for DynamoDB Repository Pattern Implementation.

Following TDD and Clean Code principles:
- Test behavior, not implementation
- One assertion per test when practical
- Fast, Independent, Repeatable, Self-validating, Timely (FIRST)
- Mock external dependencies
- Aim for 80%+ coverage
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from botocore.exceptions import ClientError
import pytest

from clarity.services.dynamodb_connection import DynamoDBConnection
from clarity.services.dynamodb_repository import (
    AuditLogRepository,
    BaseRepository,
    HealthDataRepository,
    IRepository,
    MLModelRepository,
    ProcessingJobRepository,
    RepositoryFactory,
    UserProfileRepository,
)


class TestIRepository:
    """Test the abstract repository interface."""

    def test_interface_cannot_be_instantiated(self) -> None:
        """Test that IRepository cannot be instantiated directly."""
        with pytest.raises(TypeError):
            IRepository()  # type: ignore

    def test_interface_defines_required_methods(self) -> None:
        """Test that IRepository defines all required abstract methods."""
        abstract_methods = IRepository.__abstractmethods__
        expected_methods = {"create", "get", "update", "delete"}
        assert abstract_methods == expected_methods


class TestBaseRepository:
    """Test the base repository implementation."""

    @pytest.fixture
    def mock_connection(self) -> MagicMock:
        """Create a mock DynamoDB connection."""
        return MagicMock(spec=DynamoDBConnection)

    @pytest.fixture
    def mock_table(self) -> MagicMock:
        """Create a mock DynamoDB table."""
        table = MagicMock()
        table.put_item = MagicMock()
        table.get_item = MagicMock()
        table.update_item = MagicMock()
        table.delete_item = MagicMock()
        table.batch_writer = MagicMock()
        return table

    @pytest.fixture
    def base_repository(
        self, mock_connection: MagicMock, mock_table: MagicMock
    ) -> BaseRepository:
        """Create a BaseRepository instance with mocks."""
        repo = BaseRepository(mock_connection, "test_table")
        # Mock the resource and table
        mock_resource = MagicMock()
        mock_resource.Table.return_value = mock_table
        mock_connection.get_resource.return_value = mock_resource
        return repo

    @pytest.mark.asyncio
    async def test_create_adds_timestamps_and_returns_id(
        self, base_repository: BaseRepository, mock_table: MagicMock
    ) -> None:
        """Test that create adds timestamps and returns the entity ID."""
        # Arrange
        entity = {"id": "test-id-123", "name": "Test Entity"}

        # Act
        with patch("clarity.services.dynamodb_repository.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = (
                "2024-01-01T00:00:00Z"
            )
            result = await base_repository.create(entity)

        # Assert
        assert result == "test-id-123"
        mock_table.put_item.assert_called_once()
        call_args = mock_table.put_item.call_args[1]["Item"]
        assert call_args["created_at"] == "2024-01-01T00:00:00Z"
        assert call_args["updated_at"] == "2024-01-01T00:00:00Z"

    @pytest.mark.asyncio
    async def test_create_handles_dynamodb_errors(
        self, base_repository: BaseRepository, mock_table: MagicMock
    ) -> None:
        """Test that create propagates DynamoDB errors."""
        # Arrange
        entity = {"id": "test-id", "data": "test"}
        mock_table.put_item.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}}, "PutItem"
        )

        # Act & Assert
        with pytest.raises(ClientError):
            await base_repository.create(entity)

    @pytest.mark.asyncio
    async def test_get_returns_item_when_exists(
        self, base_repository: BaseRepository, mock_table: MagicMock
    ) -> None:
        """Test that get returns the item when it exists."""
        # Arrange
        mock_table.get_item.return_value = {
            "Item": {"id": "test-id", "name": "Test Item"}
        }

        # Act
        result = await base_repository.get("test-id")

        # Assert
        assert result == {"id": "test-id", "name": "Test Item"}
        mock_table.get_item.assert_called_once_with(Key={"id": "test-id"})

    @pytest.mark.asyncio
    async def test_get_returns_none_when_not_exists(
        self, base_repository: BaseRepository, mock_table: MagicMock
    ) -> None:
        """Test that get returns None when item doesn't exist."""
        # Arrange
        mock_table.get_item.return_value = {}  # No 'Item' key

        # Act
        result = await base_repository.get("non-existent")

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_update_builds_expression_and_updates_timestamp(
        self, base_repository: BaseRepository, mock_table: MagicMock
    ) -> None:
        """Test that update builds correct expression and updates timestamp."""
        # Arrange
        updates = {"status": "active", "count": 5}

        # Act
        with patch("clarity.services.dynamodb_repository.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = (
                "2024-01-01T12:00:00Z"
            )
            result = await base_repository.update("test-id", updates)

        # Assert
        assert result is True
        mock_table.update_item.assert_called_once()
        call_args = mock_table.update_item.call_args[1]
        assert call_args["Key"] == {"id": "test-id"}
        # Check that update expression includes all fields plus updated_at
        assert "status = :status" in call_args["UpdateExpression"]
        assert "count = :count" in call_args["UpdateExpression"]
        assert "updated_at = :updated_at" in call_args["UpdateExpression"]
        assert (
            call_args["ExpressionAttributeValues"][":updated_at"]
            == "2024-01-01T12:00:00Z"
        )

    @pytest.mark.asyncio
    async def test_update_returns_false_on_error(
        self, base_repository: BaseRepository, mock_table: MagicMock
    ) -> None:
        """Test that update returns False on error."""
        # Arrange
        mock_table.update_item.side_effect = Exception("Update failed")

        # Act
        result = await base_repository.update("test-id", {"status": "failed"})

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_removes_item_successfully(
        self, base_repository: BaseRepository, mock_table: MagicMock
    ) -> None:
        """Test that delete removes item successfully."""
        # Act
        result = await base_repository.delete("test-id")

        # Assert
        assert result is True
        mock_table.delete_item.assert_called_once_with(Key={"id": "test-id"})

    @pytest.mark.asyncio
    async def test_delete_returns_false_on_error(
        self, base_repository: BaseRepository, mock_table: MagicMock
    ) -> None:
        """Test that delete returns False on error."""
        # Arrange
        mock_table.delete_item.side_effect = Exception("Delete failed")

        # Act
        result = await base_repository.delete("test-id")

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_batch_create_handles_large_batches(
        self, base_repository: BaseRepository, mock_table: MagicMock
    ) -> None:
        """Test that batch_create handles batches larger than 25 items."""
        # Arrange
        entities = [{"id": f"test-{i}", "data": f"data-{i}"} for i in range(50)]

        # Setup batch writer mock
        batch_writer = MagicMock()
        batch_writer.__enter__ = MagicMock(return_value=batch_writer)
        batch_writer.__exit__ = MagicMock(return_value=None)
        mock_table.batch_writer.return_value = batch_writer

        # Act
        with patch("clarity.services.dynamodb_repository.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = (
                "2024-01-01T00:00:00Z"
            )
            await base_repository.batch_create(entities)

        # Assert
        # Should be called twice (25 + 25 items)
        assert mock_table.batch_writer.call_count == 2
        # All items should have been written
        assert batch_writer.put_item.call_count == 50

    @pytest.mark.asyncio
    async def test_batch_create_adds_timestamps(
        self, base_repository: BaseRepository, mock_table: MagicMock
    ) -> None:
        """Test that batch_create adds timestamps to entities."""
        # Arrange
        entities = [{"id": "1", "data": "test"}, {"id": "2", "data": "test2"}]

        batch_writer = MagicMock()
        batch_writer.__enter__ = MagicMock(return_value=batch_writer)
        batch_writer.__exit__ = MagicMock(return_value=None)
        mock_table.batch_writer.return_value = batch_writer

        # Act
        with patch("clarity.services.dynamodb_repository.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = (
                "2024-01-01T00:00:00Z"
            )
            await base_repository.batch_create(entities)

        # Assert
        # Check that timestamps were added
        for call in batch_writer.put_item.call_args_list:
            item = call[1]["Item"]
            assert item["created_at"] == "2024-01-01T00:00:00Z"
            assert item["updated_at"] == "2024-01-01T00:00:00Z"


class TestHealthDataRepository:
    """Test HealthDataRepository specific functionality."""

    @pytest.fixture
    def mock_connection(self) -> MagicMock:
        """Create a mock DynamoDB connection."""
        return MagicMock(spec=DynamoDBConnection)

    @pytest.fixture
    def mock_table(self) -> MagicMock:
        """Create a mock DynamoDB table."""
        table = MagicMock()
        table.query = MagicMock()
        return table

    @pytest.fixture
    def health_repo(
        self, mock_connection: MagicMock, mock_table: MagicMock
    ) -> HealthDataRepository:
        """Create a HealthDataRepository instance with mocks."""
        repo = HealthDataRepository(mock_connection)
        # Mock the resource and table
        mock_resource = MagicMock()
        mock_resource.Table.return_value = mock_table
        mock_connection.get_resource.return_value = mock_resource
        return repo

    @pytest.mark.asyncio
    async def test_get_by_user_queries_with_correct_parameters(
        self, health_repo: HealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test that get_by_user queries with correct parameters."""
        # Arrange
        user_id = "user-123"
        mock_table.query.return_value = {
            "Items": [
                {"id": "1", "user_id": user_id, "metric": "heart_rate"},
                {"id": "2", "user_id": user_id, "metric": "steps"},
            ]
        }

        # Act
        result = await health_repo.get_by_user(user_id, limit=50)

        # Assert
        assert len(result) == 2
        mock_table.query.assert_called_once_with(
            KeyConditionExpression="user_id = :user_id",
            ExpressionAttributeValues={":user_id": user_id},
            Limit=50,
            ScanIndexForward=False,
        )

    @pytest.mark.asyncio
    async def test_get_by_user_handles_errors_gracefully(
        self, health_repo: HealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test that get_by_user returns empty list on error."""
        # Arrange
        mock_table.query.side_effect = Exception("Query failed")

        # Act
        result = await health_repo.get_by_user("user-123")

        # Assert
        assert result == []

    @pytest.mark.asyncio
    async def test_get_by_date_range_filters_correctly(
        self, health_repo: HealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test that get_by_date_range applies correct filters."""
        # Arrange
        user_id = "user-123"
        start_date = datetime(2024, 1, 1, tzinfo=UTC)
        end_date = datetime(2024, 1, 31, tzinfo=UTC)

        mock_table.query.return_value = {"Items": [{"id": "1", "data": "test"}]}

        # Act
        result = await health_repo.get_by_date_range(user_id, start_date, end_date)

        # Assert
        mock_table.query.assert_called_once_with(
            KeyConditionExpression="user_id = :user_id",
            FilterExpression="created_at BETWEEN :start AND :end",
            ExpressionAttributeValues={
                ":user_id": user_id,
                ":start": start_date.isoformat(),
                ":end": end_date.isoformat(),
            },
        )

    @pytest.mark.asyncio
    async def test_get_by_date_range_handles_errors(
        self, health_repo: HealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test that get_by_date_range returns empty list on error."""
        # Arrange
        mock_table.query.side_effect = Exception("Date range query failed")

        # Act
        result = await health_repo.get_by_date_range(
            "user-123",
            datetime.now(UTC),
            datetime.now(UTC),
        )

        # Assert
        assert result == []


class TestProcessingJobRepository:
    """Test ProcessingJobRepository specific functionality."""

    @pytest.fixture
    def mock_connection(self) -> MagicMock:
        """Create a mock DynamoDB connection."""
        return MagicMock(spec=DynamoDBConnection)

    @pytest.fixture
    def mock_table(self) -> MagicMock:
        """Create a mock DynamoDB table."""
        table = MagicMock()
        table.scan = MagicMock()
        table.update_item = MagicMock()
        return table

    @pytest.fixture
    def job_repo(
        self, mock_connection: MagicMock, mock_table: MagicMock
    ) -> ProcessingJobRepository:
        """Create a ProcessingJobRepository instance with mocks."""
        repo = ProcessingJobRepository(mock_connection)
        # Mock the resource and table
        mock_resource = MagicMock()
        mock_resource.Table.return_value = mock_table
        mock_connection.get_resource.return_value = mock_resource
        return repo

    @pytest.mark.asyncio
    async def test_get_by_status_scans_with_filter(
        self, job_repo: ProcessingJobRepository, mock_table: MagicMock
    ) -> None:
        """Test that get_by_status scans with correct filter."""
        # Arrange
        status = "processing"
        mock_table.scan.return_value = {
            "Items": [
                {"id": "job1", "status": status},
                {"id": "job2", "status": status},
            ]
        }

        # Act
        result = await job_repo.get_by_status(status)

        # Assert
        assert len(result) == 2
        mock_table.scan.assert_called_once_with(
            FilterExpression="status = :status",
            ExpressionAttributeValues={":status": status},
        )

    @pytest.mark.asyncio
    async def test_get_by_status_handles_errors(
        self, job_repo: ProcessingJobRepository, mock_table: MagicMock
    ) -> None:
        """Test that get_by_status returns empty list on error."""
        # Arrange
        mock_table.scan.side_effect = Exception("Scan failed")

        # Act
        result = await job_repo.get_by_status("failed")

        # Assert
        assert result == []

    @pytest.mark.asyncio
    async def test_update_status_updates_correctly(
        self, job_repo: ProcessingJobRepository
    ) -> None:
        """Test that update_status calls update with correct parameters."""
        # Arrange
        job_id = "job-123"
        status = "completed"
        progress = 100

        # Mock the parent update method
        job_repo.update = AsyncMock(return_value=True)

        # Act
        result = await job_repo.update_status(job_id, status, progress)

        # Assert
        assert result is True
        job_repo.update.assert_called_once_with(
            job_id, {"status": status, "progress": progress}
        )


class TestUserProfileRepository:
    """Test UserProfileRepository specific functionality."""

    @pytest.fixture
    def mock_connection(self) -> MagicMock:
        """Create a mock DynamoDB connection."""
        return MagicMock(spec=DynamoDBConnection)

    @pytest.fixture
    def mock_table(self) -> MagicMock:
        """Create a mock DynamoDB table."""
        table = MagicMock()
        table.scan = MagicMock()
        return table

    @pytest.fixture
    def user_repo(
        self, mock_connection: MagicMock, mock_table: MagicMock
    ) -> UserProfileRepository:
        """Create a UserProfileRepository instance with mocks."""
        repo = UserProfileRepository(mock_connection)
        # Mock the resource and table
        mock_resource = MagicMock()
        mock_resource.Table.return_value = mock_table
        mock_connection.get_resource.return_value = mock_resource
        return repo

    @pytest.mark.asyncio
    async def test_get_by_email_returns_first_match(
        self, user_repo: UserProfileRepository, mock_table: MagicMock
    ) -> None:
        """Test that get_by_email returns the first matching user."""
        # Arrange
        email = "test@example.com"
        mock_table.scan.return_value = {
            "Items": [
                {"id": "user1", "email": email, "name": "Test User"},
                {
                    "id": "user2",
                    "email": email,
                    "name": "Duplicate",
                },  # Shouldn't happen
            ]
        }

        # Act
        result = await user_repo.get_by_email(email)

        # Assert
        assert result == {"id": "user1", "email": email, "name": "Test User"}
        mock_table.scan.assert_called_once_with(
            FilterExpression="email = :email",
            ExpressionAttributeValues={":email": email},
        )

    @pytest.mark.asyncio
    async def test_get_by_email_returns_none_when_not_found(
        self, user_repo: UserProfileRepository, mock_table: MagicMock
    ) -> None:
        """Test that get_by_email returns None when no user found."""
        # Arrange
        mock_table.scan.return_value = {"Items": []}

        # Act
        result = await user_repo.get_by_email("notfound@example.com")

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_email_handles_errors(
        self, user_repo: UserProfileRepository, mock_table: MagicMock
    ) -> None:
        """Test that get_by_email returns None on error."""
        # Arrange
        mock_table.scan.side_effect = Exception("Scan failed")

        # Act
        result = await user_repo.get_by_email("error@example.com")

        # Assert
        assert result is None


class TestAuditLogRepository:
    """Test AuditLogRepository specific functionality."""

    @pytest.fixture
    def mock_connection(self) -> MagicMock:
        """Create a mock DynamoDB connection."""
        return MagicMock(spec=DynamoDBConnection)

    @pytest.fixture
    def mock_table(self) -> MagicMock:
        """Create a mock DynamoDB table."""
        table = MagicMock()
        table.scan = MagicMock()
        return table

    @pytest.fixture
    def audit_repo(
        self, mock_connection: MagicMock, mock_table: MagicMock
    ) -> AuditLogRepository:
        """Create an AuditLogRepository instance with mocks."""
        repo = AuditLogRepository(mock_connection)
        # Mock the resource and table
        mock_resource = MagicMock()
        mock_resource.Table.return_value = mock_table
        mock_connection.get_resource.return_value = mock_resource
        return repo

    @pytest.mark.asyncio
    async def test_create_audit_log_generates_uuid_and_calls_create(
        self, audit_repo: AuditLogRepository
    ) -> None:
        """Test that create_audit_log generates UUID and calls parent create."""
        # Arrange
        test_uuid = uuid.uuid4()
        audit_repo.create = AsyncMock(return_value=str(test_uuid))

        # Act
        with patch("uuid.uuid4", return_value=test_uuid):
            result = await audit_repo.create_audit_log(
                operation="UPDATE",
                table="test_table",
                item_id="item-123",
                user_id="user-456",
                metadata={"field": "value"},
            )

        # Assert
        assert result == str(test_uuid)
        audit_repo.create.assert_called_once()
        call_args = audit_repo.create.call_args[0][0]
        assert call_args["id"] == str(test_uuid)
        assert call_args["operation"] == "UPDATE"
        assert call_args["table"] == "test_table"
        assert call_args["item_id"] == "item-123"
        assert call_args["user_id"] == "user-456"
        assert call_args["metadata"] == {"field": "value"}
        assert call_args["source"] == "dynamodb_repository"

    @pytest.mark.asyncio
    async def test_create_audit_log_handles_optional_parameters(
        self, audit_repo: AuditLogRepository
    ) -> None:
        """Test that create_audit_log handles optional parameters correctly."""
        # Arrange
        audit_repo.create = AsyncMock(return_value="audit-id")

        # Act
        result = await audit_repo.create_audit_log(
            operation="DELETE",
            table="test_table",
            item_id="item-123",
        )

        # Assert
        call_args = audit_repo.create.call_args[0][0]
        assert call_args["user_id"] is None
        assert call_args["metadata"] == {}

    @pytest.mark.asyncio
    async def test_get_by_item_scans_with_filter(
        self, audit_repo: AuditLogRepository, mock_table: MagicMock
    ) -> None:
        """Test that get_by_item scans with correct filter."""
        # Arrange
        item_id = "item-123"
        mock_table.scan.return_value = {
            "Items": [
                {"id": "audit1", "item_id": item_id, "operation": "CREATE"},
                {"id": "audit2", "item_id": item_id, "operation": "UPDATE"},
            ]
        }

        # Act
        result = await audit_repo.get_by_item(item_id)

        # Assert
        assert len(result) == 2
        mock_table.scan.assert_called_once_with(
            FilterExpression="item_id = :item_id",
            ExpressionAttributeValues={":item_id": item_id},
        )

    @pytest.mark.asyncio
    async def test_get_by_item_handles_errors(
        self, audit_repo: AuditLogRepository, mock_table: MagicMock
    ) -> None:
        """Test that get_by_item returns empty list on error."""
        # Arrange
        mock_table.scan.side_effect = Exception("Scan failed")

        # Act
        result = await audit_repo.get_by_item("item-error")

        # Assert
        assert result == []


class TestMLModelRepository:
    """Test MLModelRepository specific functionality."""

    @pytest.fixture
    def mock_connection(self) -> MagicMock:
        """Create a mock DynamoDB connection."""
        return MagicMock(spec=DynamoDBConnection)

    @pytest.fixture
    def mock_table(self) -> MagicMock:
        """Create a mock DynamoDB table."""
        table = MagicMock()
        table.query = MagicMock()
        return table

    @pytest.fixture
    def ml_repo(
        self, mock_connection: MagicMock, mock_table: MagicMock
    ) -> MLModelRepository:
        """Create an MLModelRepository instance with mocks."""
        repo = MLModelRepository(mock_connection)
        # Mock the resource and table
        mock_resource = MagicMock()
        mock_resource.Table.return_value = mock_table
        mock_connection.get_resource.return_value = mock_resource
        return repo

    @pytest.mark.asyncio
    async def test_get_latest_version_queries_correctly(
        self, ml_repo: MLModelRepository, mock_table: MagicMock
    ) -> None:
        """Test that get_latest_version queries with correct parameters."""
        # Arrange
        model_type = "classification"
        mock_table.query.return_value = {
            "Items": [
                {
                    "id": "model-v2",
                    "model_type": model_type,
                    "version": "2.0.0",
                    "created_at": "2024-01-02",
                }
            ]
        }

        # Act
        result = await ml_repo.get_latest_version(model_type)

        # Assert
        assert result == {
            "id": "model-v2",
            "model_type": model_type,
            "version": "2.0.0",
            "created_at": "2024-01-02",
        }
        mock_table.query.assert_called_once_with(
            KeyConditionExpression="model_type = :type",
            ExpressionAttributeValues={":model_type": model_type},
            ScanIndexForward=False,
            Limit=1,
        )

    @pytest.mark.asyncio
    async def test_get_latest_version_returns_none_when_not_found(
        self, ml_repo: MLModelRepository, mock_table: MagicMock
    ) -> None:
        """Test that get_latest_version returns None when no model found."""
        # Arrange
        mock_table.query.return_value = {"Items": []}

        # Act
        result = await ml_repo.get_latest_version("nonexistent")

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_get_latest_version_handles_errors(
        self, ml_repo: MLModelRepository, mock_table: MagicMock
    ) -> None:
        """Test that get_latest_version returns None on error."""
        # Arrange
        mock_table.query.side_effect = Exception("Query failed")

        # Act
        result = await ml_repo.get_latest_version("error-type")

        # Assert
        assert result is None


class TestRepositoryFactory:
    """Test RepositoryFactory functionality."""

    @pytest.fixture
    def mock_connection(self) -> MagicMock:
        """Create a mock DynamoDB connection."""
        return MagicMock(spec=DynamoDBConnection)

    @pytest.fixture
    def factory(self, mock_connection: MagicMock) -> RepositoryFactory:
        """Create a RepositoryFactory instance."""
        return RepositoryFactory(mock_connection)

    def test_get_health_data_repository_returns_singleton(
        self, factory: RepositoryFactory
    ) -> None:
        """Test that get_health_data_repository returns the same instance."""
        # Act
        repo1 = factory.get_health_data_repository()
        repo2 = factory.get_health_data_repository()

        # Assert
        assert repo1 is repo2
        assert isinstance(repo1, HealthDataRepository)

    def test_get_processing_job_repository_returns_singleton(
        self, factory: RepositoryFactory
    ) -> None:
        """Test that get_processing_job_repository returns the same instance."""
        # Act
        repo1 = factory.get_processing_job_repository()
        repo2 = factory.get_processing_job_repository()

        # Assert
        assert repo1 is repo2
        assert isinstance(repo1, ProcessingJobRepository)

    def test_get_user_profile_repository_returns_singleton(
        self, factory: RepositoryFactory
    ) -> None:
        """Test that get_user_profile_repository returns the same instance."""
        # Act
        repo1 = factory.get_user_profile_repository()
        repo2 = factory.get_user_profile_repository()

        # Assert
        assert repo1 is repo2
        assert isinstance(repo1, UserProfileRepository)

    def test_get_audit_log_repository_returns_singleton(
        self, factory: RepositoryFactory
    ) -> None:
        """Test that get_audit_log_repository returns the same instance."""
        # Act
        repo1 = factory.get_audit_log_repository()
        repo2 = factory.get_audit_log_repository()

        # Assert
        assert repo1 is repo2
        assert isinstance(repo1, AuditLogRepository)

    def test_get_ml_model_repository_returns_singleton(
        self, factory: RepositoryFactory
    ) -> None:
        """Test that get_ml_model_repository returns the same instance."""
        # Act
        repo1 = factory.get_ml_model_repository()
        repo2 = factory.get_ml_model_repository()

        # Assert
        assert repo1 is repo2
        assert isinstance(repo1, MLModelRepository)

    def test_factory_creates_different_repository_types(
        self, factory: RepositoryFactory
    ) -> None:
        """Test that factory creates different repository types correctly."""
        # Act
        health_repo = factory.get_health_data_repository()
        job_repo = factory.get_processing_job_repository()
        user_repo = factory.get_user_profile_repository()
        audit_repo = factory.get_audit_log_repository()
        ml_repo = factory.get_ml_model_repository()

        # Assert - all should be different instances of different types
        repos = [health_repo, job_repo, user_repo, audit_repo, ml_repo]
        types = [type(repo) for repo in repos]

        # All should be different types
        assert len(set(types)) == len(types)

        # Verify correct types
        assert isinstance(health_repo, HealthDataRepository)
        assert isinstance(job_repo, ProcessingJobRepository)
        assert isinstance(user_repo, UserProfileRepository)
        assert isinstance(audit_repo, AuditLogRepository)
        assert isinstance(ml_repo, MLModelRepository)


class TestErrorScenarios:
    """Test error handling scenarios across repositories."""

    @pytest.fixture
    def mock_connection(self) -> MagicMock:
        """Create a mock DynamoDB connection."""
        connection = MagicMock(spec=DynamoDBConnection)
        # Make get_resource raise an error
        connection.get_resource.side_effect = ClientError(
            {"Error": {"Code": "ServiceUnavailable"}}, "DescribeTable"
        )
        return connection

    @pytest.mark.asyncio
    async def test_base_repository_handles_connection_errors(
        self, mock_connection: MagicMock
    ) -> None:
        """Test that repositories handle connection errors properly."""
        # Arrange
        repo = BaseRepository(mock_connection, "test_table")

        # Act & Assert
        with pytest.raises(ClientError):
            await repo.create({"id": "test", "data": "value"})

    @pytest.mark.asyncio
    async def test_batch_create_handles_batch_writer_errors(
        self, mock_connection: MagicMock
    ) -> None:
        """Test that batch_create handles batch writer errors."""
        # Arrange
        repo = BaseRepository(mock_connection, "test_table")

        # Mock successful connection but failed batch writer
        mock_resource = MagicMock()
        mock_table = MagicMock()
        mock_batch_writer = MagicMock()
        mock_batch_writer.__enter__.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}}, "BatchWriteItem"
        )
        mock_table.batch_writer.return_value = mock_batch_writer
        mock_resource.Table.return_value = mock_table
        mock_connection.get_resource.return_value = mock_resource

        # Act & Assert
        with pytest.raises(ClientError):
            await repo.batch_create([{"id": "1", "data": "test"}])


class TestAsyncBehavior:
    """Test async behavior and concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_operations_use_same_connection(
        self,
    ) -> None:
        """Test that concurrent operations share the same connection."""
        # Arrange
        mock_connection = MagicMock(spec=DynamoDBConnection)
        mock_resource = MagicMock()
        mock_table = MagicMock()

        # Setup async returns
        mock_table.get_item = MagicMock(
            return_value={"Item": {"id": "test", "data": "value"}}
        )
        mock_resource.Table.return_value = mock_table
        mock_connection.get_resource.return_value = mock_resource

        repo = BaseRepository(mock_connection, "test_table")

        # Act - perform multiple concurrent operations
        results = await asyncio.gather(
            repo.get("id1"),
            repo.get("id2"),
            repo.get("id3"),
        )

        # Assert
        assert len(results) == 3
        # Connection should be retrieved only once
        assert mock_connection.get_resource.call_count == 1

    @pytest.mark.asyncio
    async def test_async_operations_run_in_executor(
        self,
    ) -> None:
        """Test that sync DynamoDB operations run in executor."""
        # Arrange
        mock_connection = MagicMock(spec=DynamoDBConnection)
        mock_resource = MagicMock()
        mock_table = MagicMock()
        mock_resource.Table.return_value = mock_table
        mock_connection.get_resource.return_value = mock_resource

        repo = BaseRepository(mock_connection, "test_table")

        # Mock the event loop's run_in_executor
        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_run_in_executor = AsyncMock(return_value={"Item": {"id": "test"}})
            mock_loop.run_in_executor = mock_run_in_executor
            mock_get_loop.return_value = mock_loop

            # Act
            result = await repo.get("test-id")

            # Assert
            assert result == {"id": "test"}
            # Verify run_in_executor was called
            mock_run_in_executor.assert_called_once()
            # First arg should be None (default executor)
            assert mock_run_in_executor.call_args[0][0] is None
