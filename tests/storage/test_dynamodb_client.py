"""Comprehensive tests for DynamoDB client."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import math
from unittest.mock import MagicMock, patch
import uuid

from botocore.exceptions import ClientError
import pytest

from clarity.core.exceptions import ServiceError
from clarity.models.health_data import (
    ActivityData,
    BiometricData,
    HealthMetric,
    HealthMetricType,
    MentalHealthIndicator,
    ProcessingStatus,
    SleepData,
)
from clarity.storage.dynamodb_client import DynamoDBHealthDataRepository


@pytest.fixture
def mock_table() -> MagicMock:
    """Mock DynamoDB table."""
    table = MagicMock()
    table.put_item = MagicMock()
    table.get_item = MagicMock()
    table.query = MagicMock()
    table.delete_item = MagicMock()
    table.batch_writer = MagicMock()
    return table


@pytest.fixture
def mock_dynamodb_resource(mock_table: MagicMock) -> MagicMock:
    """Mock DynamoDB resource."""
    resource = MagicMock()
    resource.Table.return_value = mock_table
    return resource


@pytest.fixture
def dynamodb_repository(
    mock_dynamodb_resource: MagicMock,
) -> DynamoDBHealthDataRepository:
    """Create DynamoDB repository with mocked resource."""
    with patch("boto3.resource", return_value=mock_dynamodb_resource):
        return DynamoDBHealthDataRepository(
            table_name="test-health-data",
            region="us-east-1",
        )


@pytest.fixture
def dynamodb_repository_with_endpoint(
    mock_dynamodb_resource: MagicMock,
) -> DynamoDBHealthDataRepository:
    """Create DynamoDB repository with endpoint URL."""
    with patch("boto3.resource", return_value=mock_dynamodb_resource):
        return DynamoDBHealthDataRepository(
            table_name="test-health-data",
            region="us-east-1",
            endpoint_url="http://localhost:8000",
        )


@pytest.fixture
def valid_health_metrics() -> list[HealthMetric]:
    """Create valid health metrics."""
    return [
        HealthMetric(
            metric_id=uuid.uuid4(),
            metric_type=HealthMetricType.HEART_RATE,
            created_at=datetime.now(UTC),
            device_id="device-123",
            biometric_data=BiometricData(
                heart_rate=72,
                systolic_bp=120,
                diastolic_bp=80,
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
                active_energy=250,  # Changed from calories_burned to active_energy
            ),
        ),
        HealthMetric(
            metric_id=uuid.uuid4(),
            metric_type=HealthMetricType.SLEEP_ANALYSIS,
            created_at=datetime.now(UTC),
            device_id="device-456",
            sleep_data=SleepData(
                total_sleep_minutes=450,  # 7.5 hours
                sleep_start=datetime.now(UTC) - timedelta(hours=8),
                sleep_end=datetime.now(UTC) - timedelta(minutes=30),
                duration_hours=7.5,
                deep_sleep_hours=2.0,
                rem_sleep_hours=1.5,
                light_sleep_hours=4.0,
                awake_hours=0.5,
                sleep_efficiency=0.88,
            ),
        ),
        HealthMetric(
            metric_id=uuid.uuid4(),
            metric_type=HealthMetricType.MOOD_ASSESSMENT,
            created_at=datetime.now(UTC),
            device_id="device-789",
            mental_health_data=MentalHealthIndicator(
                mood_score="good",  # Changed from numeric to MoodScale enum value
                stress_level=3,
                anxiety_level=2,
            ),
            raw_data={"mood_notes": "Feeling good"},
            metadata={"assessment_version": "2.0"},
        ),
    ]


class TestDynamoDBHealthDataRepositoryInit:
    """Test repository initialization."""

    def test_init_without_endpoint(self) -> None:
        """Test initialization without endpoint URL."""
        with patch("boto3.resource") as mock_boto_resource:
            repo = DynamoDBHealthDataRepository(
                table_name="test-table",
                region="us-west-2",
            )

            assert repo.table_name == "test-table"
            assert repo.region == "us-west-2"
            mock_boto_resource.assert_called_once_with(
                "dynamodb", region_name="us-west-2"
            )

    def test_init_with_endpoint(self) -> None:
        """Test initialization with endpoint URL."""
        with patch("boto3.resource") as mock_boto_resource:
            repo = DynamoDBHealthDataRepository(
                table_name="test-table",
                region="us-east-1",
                endpoint_url="http://localhost:8000",
            )

            assert repo.table_name == "test-table"
            assert repo.region == "us-east-1"
            mock_boto_resource.assert_called_once_with(
                "dynamodb",
                region_name="us-east-1",
                endpoint_url="http://localhost:8000",
            )


class TestSerializationMethods:
    """Test serialization and deserialization methods."""

    def test_serialize_item_with_floats(
        self, dynamodb_repository: DynamoDBHealthDataRepository
    ) -> None:
        """Test serialization converts floats to Decimal."""
        data = {
            "score": 0.95,
            "nested": {"value": 1.23},
            "list": [1.1, 2.2, 3.3],
        }

        result = dynamodb_repository._serialize_item(data)

        assert isinstance(result["score"], Decimal)
        assert result["score"] == Decimal("0.95")
        assert isinstance(result["nested"]["value"], Decimal)
        assert all(isinstance(v, Decimal) for v in result["list"])

    def test_serialize_item_with_datetime(
        self, dynamodb_repository: DynamoDBHealthDataRepository
    ) -> None:
        """Test serialization converts datetime to ISO string."""
        now = datetime.now(UTC)
        data = {"created_at": now, "metadata": {"timestamp": now}}

        result = dynamodb_repository._serialize_item(data)

        assert result["created_at"] == now.isoformat()
        assert result["metadata"]["timestamp"] == now.isoformat()

    def test_serialize_item_mixed_types(
        self, dynamodb_repository: DynamoDBHealthDataRepository
    ) -> None:
        """Test serialization with mixed types."""
        data = {
            "string": "test",
            "int": 42,
            "float": math.pi,
            "bool": True,
            "none": None,
            "list": ["a", 1, 2.5],
            "dict": {"nested": 1.5},
        }

        result = dynamodb_repository._serialize_item(data)

        assert result["string"] == "test"
        assert result["int"] == 42
        assert isinstance(result["float"], Decimal)
        assert result["bool"] is True
        assert result["none"] is None
        assert result["list"][2] == Decimal("2.5")
        assert isinstance(result["dict"]["nested"], Decimal)

    def test_deserialize_item_with_decimals(
        self, dynamodb_repository: DynamoDBHealthDataRepository
    ) -> None:
        """Test deserialization converts Decimal to float."""
        data = {
            "score": Decimal("0.95"),
            "nested": {"value": Decimal("1.23")},
            "list": [Decimal("1.1"), Decimal("2.2")],
        }

        result = dynamodb_repository._deserialize_item(data)

        assert isinstance(result["score"], float)
        assert result["score"] == 0.95
        assert isinstance(result["nested"]["value"], float)
        assert all(isinstance(v, float) for v in result["list"])

    def test_deserialize_item_mixed_types(
        self, dynamodb_repository: DynamoDBHealthDataRepository
    ) -> None:
        """Test deserialization with mixed types."""
        data = {
            "string": "test",
            "decimal": Decimal("3.14"),
            "int": 42,
            "list": ["a", Decimal("1.5")],
            "dict": {"nested": Decimal("2.5")},
        }

        result = dynamodb_repository._deserialize_item(data)

        assert result["string"] == "test"
        assert isinstance(result["decimal"], float)
        assert result["decimal"] == 3.14  # noqa: FURB152
        assert result["int"] == 42
        assert result["list"][1] == 1.5
        assert result["dict"]["nested"] == 2.5


class TestSaveHealthData:
    """Test save_health_data method."""

    @pytest.mark.asyncio
    async def test_save_health_data_success(
        self,
        dynamodb_repository: DynamoDBHealthDataRepository,
        mock_table: MagicMock,
        valid_health_metrics: list[HealthMetric],
    ) -> None:
        """Test successful health data save."""
        user_id = str(uuid.uuid4())
        processing_id = str(uuid.uuid4())
        upload_source = "mobile_app"
        client_timestamp = datetime.now(UTC)

        result = await dynamodb_repository.save_health_data(
            user_id=user_id,
            processing_id=processing_id,
            metrics=valid_health_metrics,
            upload_source=upload_source,
            client_timestamp=client_timestamp,
        )

        assert result is True
        mock_table.put_item.assert_called_once()

        # Check the saved item structure
        call_args = mock_table.put_item.call_args[1]
        item = call_args["Item"]

        assert item["pk"] == f"USER#{user_id}"
        assert item["sk"].startswith("HEALTH#")
        assert item["user_id"] == user_id
        assert item["processing_id"] == processing_id
        assert item["upload_source"] == upload_source
        assert item["processing_status"] == ProcessingStatus.RECEIVED.value

        # Check metrics are properly stored
        assert "heart_rate" in item["metrics"]
        assert "activity_level" in item["metrics"]
        assert "sleep_analysis" in item["metrics"]
        assert "mood_assessment" in item["metrics"]

        # Check specific metric data
        heart_rate_metric = item["metrics"]["heart_rate"]
        assert heart_rate_metric["biometric_data"]["heart_rate"] == Decimal(72)
        assert heart_rate_metric["device_id"] == "device-123"

        # Check TTL is set (90 days)
        assert "ttl" in item
        assert isinstance(item["ttl"], int)

    @pytest.mark.asyncio
    async def test_save_health_data_client_error(
        self,
        dynamodb_repository: DynamoDBHealthDataRepository,
        mock_table: MagicMock,
        valid_health_metrics: list[HealthMetric],
    ) -> None:
        """Test save with DynamoDB client error."""
        mock_table.put_item.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Invalid item"}},
            "PutItem",
        )

        with pytest.raises(ServiceError, match="Failed to save health data"):
            await dynamodb_repository.save_health_data(
                user_id="user-123",
                processing_id="proc-123",
                metrics=valid_health_metrics,
                upload_source="test",
                client_timestamp=datetime.now(UTC),
            )

    @pytest.mark.asyncio
    async def test_save_health_data_unexpected_error(
        self,
        dynamodb_repository: DynamoDBHealthDataRepository,
        mock_table: MagicMock,
        valid_health_metrics: list[HealthMetric],
    ) -> None:
        """Test save with unexpected error."""
        mock_table.put_item.side_effect = Exception("Unexpected error")

        with pytest.raises(ServiceError, match="Failed to save health data"):
            await dynamodb_repository.save_health_data(
                user_id="user-123",
                processing_id="proc-123",
                metrics=valid_health_metrics,
                upload_source="test",
                client_timestamp=datetime.now(UTC),
            )


class TestGetUserHealthData:
    """Test get_user_health_data method."""

    @pytest.mark.asyncio
    async def test_get_user_health_data_success(
        self, dynamodb_repository: DynamoDBHealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test successful health data retrieval."""
        user_id = "user-123"
        mock_table.query.return_value = {
            "Items": [
                {
                    "pk": f"USER#{user_id}",
                    "sk": "HEALTH#2024-01-15T12:00:00",
                    "metrics": {
                        "heart_rate": {"value": Decimal(72)},
                        "steps": {"value": Decimal(5000)},
                    },
                },
                {
                    "pk": f"USER#{user_id}",
                    "sk": "HEALTH#2024-01-15T11:00:00",
                    "metrics": {
                        "heart_rate": {"value": Decimal(68)},
                    },
                },
            ]
        }

        result = await dynamodb_repository.get_user_health_data(
            user_id=user_id,
            limit=10,
            offset=0,
        )

        assert len(result["data"]) == 2
        assert result["pagination"]["limit"] == 10
        assert result["pagination"]["offset"] == 0
        assert result["pagination"]["total"] == 2
        assert result["pagination"]["has_more"] is False

        # Check deserialization
        first_item = result["data"][0]
        assert first_item["metrics"]["heart_rate"]["value"] == 72.0
        assert first_item["metrics"]["steps"]["value"] == 5000.0

    @pytest.mark.asyncio
    async def test_get_user_health_data_with_date_range(
        self, dynamodb_repository: DynamoDBHealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test retrieval with date range filtering."""
        user_id = "user-123"
        start_date = datetime(2024, 1, 1, tzinfo=UTC)
        end_date = datetime(2024, 1, 31, tzinfo=UTC)

        mock_table.query.return_value = {"Items": []}

        await dynamodb_repository.get_user_health_data(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )

        # Check query was called with date range
        call_args = mock_table.query.call_args[1]
        assert "KeyConditionExpression" in call_args
        # Just verify the query was called - the actual key condition
        # is a boto3 object that we can't easily inspect
        mock_table.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_health_data_with_metric_filter(
        self, dynamodb_repository: DynamoDBHealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test retrieval with metric type filtering."""
        user_id = "user-123"
        mock_table.query.return_value = {
            "Items": [
                {
                    "pk": f"USER#{user_id}",
                    "sk": "HEALTH#2024-01-15T12:00:00",
                    "metrics": {"heart_rate": {"value": Decimal(72)}},
                },
                {
                    "pk": f"USER#{user_id}",
                    "sk": "HEALTH#2024-01-15T11:00:00",
                    "metrics": {"steps": {"value": Decimal(5000)}},
                },
            ]
        }

        result = await dynamodb_repository.get_user_health_data(
            user_id=user_id,
            metric_type="heart_rate",
        )

        # Should only return items with heart_rate metric
        assert len(result["data"]) == 1
        assert "heart_rate" in result["data"][0]["metrics"]

    @pytest.mark.asyncio
    async def test_get_user_health_data_with_pagination(
        self, dynamodb_repository: DynamoDBHealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test retrieval with pagination."""
        user_id = "user-123"
        items = [
            {
                "pk": f"USER#{user_id}",
                "sk": f"HEALTH#2024-01-15T{i:02d}:00:00",
                "metrics": {"heart_rate": {"value": Decimal(70)}},
            }
            for i in range(15)
        ]
        mock_table.query.return_value = {"Items": items}

        result = await dynamodb_repository.get_user_health_data(
            user_id=user_id,
            limit=5,
            offset=5,
        )

        assert len(result["data"]) == 5
        assert result["pagination"]["offset"] == 5
        assert result["pagination"]["has_more"] is True

    @pytest.mark.asyncio
    async def test_get_user_health_data_client_error(
        self, dynamodb_repository: DynamoDBHealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test retrieval with client error."""
        mock_table.query.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException"}},
            "Query",
        )

        with pytest.raises(ServiceError, match="Failed to retrieve health data"):
            await dynamodb_repository.get_user_health_data(user_id="user-123")


class TestGetProcessingStatus:
    """Test get_processing_status method."""

    @pytest.mark.asyncio
    async def test_get_processing_status_success(
        self, dynamodb_repository: DynamoDBHealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test successful status retrieval."""
        processing_id = "proc-123"
        user_id = "user-123"

        mock_table.query.return_value = {
            "Items": [
                {
                    "processing_id": processing_id,
                    "user_id": user_id,
                    "processing_status": "completed",
                    "created_at": "2024-01-15T12:00:00",
                    "updated_at": "2024-01-15T12:05:00",
                }
            ]
        }

        result = await dynamodb_repository.get_processing_status(processing_id, user_id)

        assert result is not None
        assert result["processing_id"] == processing_id
        assert result["status"] == "completed"
        assert result["created_at"] == "2024-01-15T12:00:00"
        assert result["updated_at"] == "2024-01-15T12:05:00"

    @pytest.mark.asyncio
    async def test_get_processing_status_not_found(
        self, dynamodb_repository: DynamoDBHealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test status retrieval when not found."""
        mock_table.query.return_value = {"Items": []}

        result = await dynamodb_repository.get_processing_status(
            "missing-123", "user-123"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_processing_status_wrong_user(
        self, dynamodb_repository: DynamoDBHealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test status retrieval with wrong user."""
        mock_table.query.return_value = {
            "Items": [
                {
                    "processing_id": "proc-123",
                    "user_id": "different-user",
                    "processing_status": "completed",
                }
            ]
        }

        result = await dynamodb_repository.get_processing_status("proc-123", "user-123")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_processing_status_client_error(
        self, dynamodb_repository: DynamoDBHealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test status retrieval with client error."""
        mock_table.query.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}},
            "Query",
        )

        result = await dynamodb_repository.get_processing_status("proc-123", "user-123")

        assert result is None


class TestDeleteHealthData:
    """Test delete_health_data method."""

    @pytest.mark.asyncio
    async def test_delete_health_data_specific_processing(
        self, dynamodb_repository: DynamoDBHealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test deletion of specific processing job."""
        processing_id = "proc-123"
        user_id = "user-123"

        mock_table.query.return_value = {
            "Items": [
                {
                    "pk": f"USER#{user_id}",
                    "sk": "HEALTH#2024-01-15T12:00:00",
                    "user_id": user_id,
                    "processing_id": processing_id,
                }
            ]
        }

        result = await dynamodb_repository.delete_health_data(user_id, processing_id)

        assert result is True
        mock_table.delete_item.assert_called_once_with(
            Key={"pk": f"USER#{user_id}", "sk": "HEALTH#2024-01-15T12:00:00"}
        )

    @pytest.mark.asyncio
    async def test_delete_health_data_all_user_data(
        self, dynamodb_repository: DynamoDBHealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test deletion of all user data."""
        user_id = "user-123"
        items = [
            {"pk": f"USER#{user_id}", "sk": f"HEALTH#2024-01-15T{i:02d}:00:00"}
            for i in range(3)
        ]
        mock_table.query.return_value = {"Items": items}

        # Mock batch writer
        mock_batch_writer = MagicMock()
        mock_batch_writer.__enter__ = MagicMock(return_value=mock_batch_writer)
        mock_batch_writer.__exit__ = MagicMock(return_value=None)
        mock_table.batch_writer.return_value = mock_batch_writer

        result = await dynamodb_repository.delete_health_data(user_id)

        assert result is True
        assert mock_batch_writer.delete_item.call_count == 3

    @pytest.mark.asyncio
    async def test_delete_health_data_client_error(
        self, dynamodb_repository: DynamoDBHealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test deletion with client error."""
        mock_table.query.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}},
            "Query",
        )

        with pytest.raises(ServiceError, match="Failed to delete health data"):
            await dynamodb_repository.delete_health_data("user-123")


class TestLegacyMethods:
    """Test legacy save_data and get_data methods."""

    @pytest.mark.asyncio
    async def test_save_data_success(
        self, dynamodb_repository: DynamoDBHealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test legacy save_data method."""
        user_id = "user-123"
        data = {"heart_rate": "72", "steps": "5000"}

        result = await dynamodb_repository.save_data(user_id, data)

        assert result.startswith(f"{user_id}#")
        mock_table.put_item.assert_called_once()

        call_args = mock_table.put_item.call_args[1]
        item = call_args["Item"]
        assert item["pk"] == f"USER#{user_id}"
        assert item["sk"].startswith("DATA#")
        assert item["data"] == data

    @pytest.mark.asyncio
    async def test_get_data_success(
        self, dynamodb_repository: DynamoDBHealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test legacy get_data method."""
        user_id = "user-123"
        mock_table.query.return_value = {
            "Items": [
                {
                    "pk": f"USER#{user_id}",
                    "sk": "DATA#2024-01-15T12:00:00",
                    "data": {"heart_rate": 72, "steps": 5000},
                }
            ]
        }

        result = await dynamodb_repository.get_data(user_id)

        assert result == {"heart_rate": "72", "steps": "5000"}

    @pytest.mark.asyncio
    async def test_get_data_no_results(
        self, dynamodb_repository: DynamoDBHealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test get_data with no results."""
        mock_table.query.return_value = {"Items": []}

        result = await dynamodb_repository.get_data("user-123")

        assert result == {}

    @pytest.mark.asyncio
    async def test_get_data_invalid_data_type(
        self, dynamodb_repository: DynamoDBHealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test get_data with invalid data type."""
        mock_table.query.return_value = {
            "Items": [
                {
                    "pk": "USER#user-123",
                    "sk": "DATA#2024-01-15T12:00:00",
                    "data": "not a dict",  # Invalid type
                }
            ]
        }

        result = await dynamodb_repository.get_data("user-123")

        assert result == {}


class TestInitializeAndCleanup:
    """Test initialize and cleanup methods."""

    @pytest.mark.asyncio
    async def test_initialize(
        self, dynamodb_repository: DynamoDBHealthDataRepository
    ) -> None:
        """Test repository initialization."""
        # Should not raise any exceptions
        await dynamodb_repository.initialize()

    @pytest.mark.asyncio
    async def test_cleanup(
        self, dynamodb_repository: DynamoDBHealthDataRepository
    ) -> None:
        """Test repository cleanup."""
        # Should not raise any exceptions
        await dynamodb_repository.cleanup()


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_save_health_data_with_null_optional_fields(
        self, dynamodb_repository: DynamoDBHealthDataRepository, mock_table: MagicMock
    ) -> None:
        """Test saving metrics with null optional fields."""
        metric = HealthMetric(
            metric_id=uuid.uuid4(),
            metric_type=HealthMetricType.HEART_RATE,
            created_at=datetime.now(UTC),
            device_id="device-123",
            biometric_data=BiometricData(heart_rate=72),
            raw_data=None,  # Null optional field
            metadata=None,  # Null optional field
        )

        result = await dynamodb_repository.save_health_data(
            user_id="user-123",
            processing_id="proc-123",
            metrics=[metric],
            upload_source="test",
            client_timestamp=datetime.now(UTC),
        )

        assert result is True

    def test_serialize_deeply_nested_structure(
        self, dynamodb_repository: DynamoDBHealthDataRepository
    ) -> None:
        """Test serialization of deeply nested structures."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "value": math.pi,
                            "list": [1.1, 2.2, {"nested": 4.4}],
                        }
                    }
                }
            }
        }

        result = dynamodb_repository._serialize_item(data)

        # Check deep nesting is preserved
        level4 = result["level1"]["level2"]["level3"]["level4"]
        assert isinstance(level4["value"], Decimal)
        assert isinstance(level4["list"][0], Decimal)
        assert isinstance(level4["list"][2]["nested"], Decimal)
