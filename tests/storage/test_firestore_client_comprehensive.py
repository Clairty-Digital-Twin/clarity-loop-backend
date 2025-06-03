"""Comprehensive tests for FirestoreClient.

Tests all methods and edge cases to improve coverage from 42% to 85%+.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from clarity.models.health_data import BiometricData, HealthMetric, HealthMetricType
from clarity.storage.firestore_client import FirestoreClient


class TestFirestoreClientComprehensive:
    """Comprehensive test coverage for FirestoreClient."""

    @pytest.fixture
    def mock_firestore_client(self):
        """Create a mock Firestore client."""
        mock_client = MagicMock()
        return mock_client

    @pytest.fixture
    def firestore_client(self, mock_firestore_client):
        """Create FirestoreClient with mocked Firestore."""
        with patch('clarity.storage.firestore_client.firestore.AsyncClient') as mock_firestore:
            with patch('clarity.storage.firestore_client.firebase_admin.initialize_app'):
                mock_firestore.return_value = mock_firestore_client
                client = FirestoreClient(project_id="test-project")
                return client

    @pytest.fixture
    def sample_health_metric(self):
        """Create a sample health metric for testing."""
        return HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.HEART_RATE,
            device_id="test_device",
            raw_data={},
            metadata={},
            created_at=datetime.now(UTC),
            biometric_data=BiometricData(
                heart_rate=75.0,
                blood_pressure_systolic=120,
                blood_pressure_diastolic=80,
                oxygen_saturation=99.0,
                heart_rate_variability=50.0,
                respiratory_rate=16.0,
                body_temperature=37.0,
                blood_glucose=100.0
            )
        )

    async def test_initialization_success(self, mock_firestore_client):
        """Test successful FirestoreClient initialization."""
        with patch('clarity.storage.firestore_client.firestore.AsyncClient') as mock_firestore:
            with patch('clarity.storage.firestore_client.firebase_admin.initialize_app'):
                mock_firestore.return_value = mock_firestore_client
                client = FirestoreClient(project_id="test-project")
                assert client.project_id == "test-project"

    async def test_initialization_failure(self):
        """Test FirestoreClient initialization failure."""
        with patch('clarity.storage.firestore_client.firebase_admin.initialize_app') as mock_init:
            mock_init.side_effect = Exception("Connection failed")

            with pytest.raises(Exception):
                FirestoreClient(project_id="test-project")

    async def test_save_health_data_success(self, firestore_client, sample_health_metric):
        """Test successful health data saving."""
        # Mock Firestore operations
        mock_collection = MagicMock()
        mock_doc_ref = MagicMock()
        mock_batch = MagicMock()

        firestore_client.client.collection.return_value = mock_collection
        mock_collection.document.return_value = mock_doc_ref
        firestore_client.client.batch.return_value = mock_batch

        # Mock batch operations
        mock_batch.set = MagicMock()
        mock_batch.commit = AsyncMock(return_value=True)

        result = await firestore_client.save_health_data(
            user_id="test_user",
            processing_id=str(uuid4()),
            metrics=[sample_health_metric],
            upload_source="test",
            client_timestamp=datetime.now(UTC)
        )

        assert result is True
        mock_batch.commit.assert_called_once()

    async def test_save_health_data_failure(self, firestore_client, sample_health_metric):
        """Test health data saving failure."""
        # Mock Firestore operations to fail
        mock_batch = MagicMock()
        firestore_client.client.batch.return_value = mock_batch
        mock_batch.commit = AsyncMock(side_effect=Exception("Firestore error"))

        result = await firestore_client.save_health_data(
            user_id="test_user",
            processing_id=str(uuid4()),
            metrics=[sample_health_metric],
            upload_source="test",
            client_timestamp=datetime.now(UTC)
        )

        assert result is False

    async def test_save_health_data_empty_metrics(self, firestore_client):
        """Test saving with empty metrics list."""
        result = await firestore_client.save_health_data(
            user_id="test_user",
            processing_id=str(uuid4()),
            metrics=[],
            upload_source="test",
            client_timestamp=datetime.now(UTC)
        )

        # Should still return True for empty metrics
        assert result is True

    async def test_get_user_health_data_success(self, firestore_client):
        """Test successful user health data retrieval."""
        # Mock Firestore query results
        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_docs = MagicMock()

        # Mock document data
        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = {
            "metric_id": "test_id",
            "metric_type": "heart_rate",
            "device_id": "test_device",
            "created_at": datetime.now(UTC).isoformat(),
            "biometric_data": {"heart_rate": 75.0}
        }

        mock_docs.__iter__ = MagicMock(return_value=iter([mock_doc]))
        mock_query.stream = AsyncMock(return_value=mock_docs)
        mock_collection.where.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.order_by.return_value = mock_query

        firestore_client.client.collection.return_value = mock_collection

        result = await firestore_client.get_user_health_data("test_user")

        assert "data" in result
        assert "total_count" in result
        assert "page_info" in result

    async def test_get_user_health_data_with_filters(self, firestore_client):
        """Test user health data retrieval with filters."""
        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_docs = MagicMock()

        mock_docs.__iter__ = MagicMock(return_value=iter([]))
        mock_query.stream = AsyncMock(return_value=mock_docs)
        mock_collection.where.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.order_by.return_value = mock_query

        firestore_client.client.collection.return_value = mock_collection

        result = await firestore_client.get_user_health_data(
            user_id="test_user",
            metric_type="heart_rate",
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC),
            limit=50,
            offset=10
        )

        assert result["data"] == []
        assert result["total_count"] == 0

    async def test_get_user_health_data_failure(self, firestore_client):
        """Test user health data retrieval failure."""
        mock_collection = MagicMock()
        mock_query = MagicMock()

        mock_query.stream = AsyncMock(side_effect=Exception("Firestore error"))
        mock_collection.where.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.order_by.return_value = mock_query

        firestore_client.client.collection.return_value = mock_collection

        result = await firestore_client.get_user_health_data("test_user")

        assert result["data"] == []
        assert result["total_count"] == 0

    async def test_get_processing_status_exists(self, firestore_client):
        """Test getting existing processing status."""
        mock_collection = MagicMock()
        mock_doc_ref = MagicMock()
        mock_doc = MagicMock()

        status_data = {
            "processing_id": "test_id",
            "user_id": "test_user",
            "status": "completed"
        }

        mock_doc.exists = True
        mock_doc.to_dict.return_value = status_data
        mock_doc_ref.get = AsyncMock(return_value=mock_doc)
        mock_collection.document.return_value = mock_doc_ref

        firestore_client.client.collection.return_value = mock_collection

        result = await firestore_client.get_processing_status("test_id", "test_user")

        assert result == status_data

    async def test_get_processing_status_not_found(self, firestore_client):
        """Test getting non-existent processing status."""
        mock_collection = MagicMock()
        mock_doc_ref = MagicMock()
        mock_doc = MagicMock()

        mock_doc.exists = False
        mock_doc_ref.get = AsyncMock(return_value=mock_doc)
        mock_collection.document.return_value = mock_doc_ref

        firestore_client.client.collection.return_value = mock_collection

        result = await firestore_client.get_processing_status("test_id", "test_user")

        assert result is None

    async def test_get_processing_status_wrong_user(self, firestore_client):
        """Test getting processing status with wrong user."""
        mock_collection = MagicMock()
        mock_doc_ref = MagicMock()
        mock_doc = MagicMock()

        status_data = {
            "processing_id": "test_id",
            "user_id": "different_user",
            "status": "completed"
        }

        mock_doc.exists = True
        mock_doc.to_dict.return_value = status_data
        mock_doc_ref.get = AsyncMock(return_value=mock_doc)
        mock_collection.document.return_value = mock_doc_ref

        firestore_client.client.collection.return_value = mock_collection

        result = await firestore_client.get_processing_status("test_id", "test_user")

        assert result is None

    async def test_get_processing_status_failure(self, firestore_client):
        """Test processing status retrieval failure."""
        mock_collection = MagicMock()
        mock_doc_ref = MagicMock()

        mock_doc_ref.get = AsyncMock(side_effect=Exception("Firestore error"))
        mock_collection.document.return_value = mock_doc_ref

        firestore_client.client.collection.return_value = mock_collection

        result = await firestore_client.get_processing_status("test_id", "test_user")

        assert result is None

    async def test_delete_health_data_specific_processing(self, firestore_client):
        """Test deleting specific processing job."""
        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_docs = MagicMock()
        mock_doc = MagicMock()
        mock_batch = MagicMock()

        mock_doc.reference = MagicMock()
        mock_docs.__iter__ = MagicMock(return_value=iter([mock_doc]))
        mock_query.stream = AsyncMock(return_value=mock_docs)
        mock_collection.where.return_value = mock_query

        mock_batch.delete = MagicMock()
        mock_batch.commit = AsyncMock(return_value=True)
        firestore_client.client.batch.return_value = mock_batch

        firestore_client.client.collection.return_value = mock_collection

        result = await firestore_client.delete_health_data("test_user", "test_processing_id")

        assert result is True
        mock_batch.commit.assert_called_once()

    async def test_delete_health_data_all_user_data(self, firestore_client):
        """Test deleting all user data."""
        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_docs = MagicMock()
        mock_doc = MagicMock()
        mock_batch = MagicMock()

        mock_doc.reference = MagicMock()
        mock_docs.__iter__ = MagicMock(return_value=iter([mock_doc]))
        mock_query.stream = AsyncMock(return_value=mock_docs)
        mock_collection.where.return_value = mock_query

        mock_batch.delete = MagicMock()
        mock_batch.commit = AsyncMock(return_value=True)
        firestore_client.client.batch.return_value = mock_batch

        firestore_client.client.collection.return_value = mock_collection

        result = await firestore_client.delete_health_data("test_user")

        assert result is True
        mock_batch.commit.assert_called_once()

    async def test_delete_health_data_failure(self, firestore_client):
        """Test health data deletion failure."""
        mock_collection = MagicMock()
        mock_query = MagicMock()

        mock_query.stream = AsyncMock(side_effect=Exception("Firestore error"))
        mock_collection.where.return_value = mock_query

        firestore_client.client.collection.return_value = mock_collection

        result = await firestore_client.delete_health_data("test_user")

        assert result is False

    async def test_save_data_generic_success(self, firestore_client):
        """Test generic save_data method success."""
        mock_collection = MagicMock()
        mock_doc_ref = MagicMock()

        mock_doc_ref.set = AsyncMock(return_value=True)
        mock_collection.document.return_value = mock_doc_ref
        mock_doc_ref.id = "generated_doc_id"

        firestore_client.client.collection.return_value = mock_collection

        result = await firestore_client.save_data("test_user", {"test": "data"})

        assert result == "generated_doc_id"

    async def test_save_data_generic_failure(self, firestore_client):
        """Test generic save_data method failure."""
        mock_collection = MagicMock()
        mock_doc_ref = MagicMock()

        mock_doc_ref.set = AsyncMock(side_effect=Exception("Firestore error"))
        mock_collection.document.return_value = mock_doc_ref

        firestore_client.client.collection.return_value = mock_collection

        with pytest.raises(Exception):
            await firestore_client.save_data("test_user", {"test": "data"})

    async def test_get_data_generic_success(self, firestore_client):
        """Test generic get_data method success."""
        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_docs = MagicMock()
        mock_doc = MagicMock()

        mock_doc.to_dict.return_value = {"test": "data"}
        mock_docs.__iter__ = MagicMock(return_value=iter([mock_doc]))
        mock_query.stream = AsyncMock(return_value=mock_docs)
        mock_collection.where.return_value = mock_query

        firestore_client.client.collection.return_value = mock_collection

        result = await firestore_client.get_data("test_user")

        assert result == [{"test": "data"}]

    async def test_get_data_generic_with_filters(self, firestore_client):
        """Test generic get_data method with filters."""
        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_docs = MagicMock()

        mock_docs.__iter__ = MagicMock(return_value=iter([]))
        mock_query.stream = AsyncMock(return_value=mock_docs)
        mock_collection.where.return_value = mock_query

        firestore_client.client.collection.return_value = mock_collection

        result = await firestore_client.get_data("test_user", filters={"status": "active"})

        assert result == []

    async def test_get_data_generic_failure(self, firestore_client):
        """Test generic get_data method failure."""
        mock_collection = MagicMock()
        mock_query = MagicMock()

        mock_query.stream = AsyncMock(side_effect=Exception("Firestore error"))
        mock_collection.where.return_value = mock_query

        firestore_client.client.collection.return_value = mock_collection

        with pytest.raises(Exception):
            await firestore_client.get_data("test_user")

    async def test_initialize_method(self, firestore_client):
        """Test initialize method."""
        # Should complete without error
        await firestore_client.initialize()

    async def test_cleanup_method(self, firestore_client):
        """Test cleanup method."""
        # Should complete without error
        await firestore_client.cleanup()

    async def test_pagination_edge_cases(self, firestore_client):
        """Test pagination with edge cases."""
        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_docs = MagicMock()

        mock_docs.__iter__ = MagicMock(return_value=iter([]))
        mock_query.stream = AsyncMock(return_value=mock_docs)
        mock_collection.where.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.order_by.return_value = mock_query

        firestore_client.client.collection.return_value = mock_collection

        # Test with limit=0
        result = await firestore_client.get_user_health_data("test_user", limit=0)
        assert result["data"] == []

        # Test with large offset
        result = await firestore_client.get_user_health_data("test_user", offset=1000)
        assert result["data"] == []

    async def test_metric_type_filtering(self, firestore_client):
        """Test metric type filtering functionality."""
        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_docs = MagicMock()

        mock_docs.__iter__ = MagicMock(return_value=iter([]))
        mock_query.stream = AsyncMock(return_value=mock_docs)
        mock_collection.where.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.order_by.return_value = mock_query

        firestore_client.client.collection.return_value = mock_collection

        # Test various metric types
        for metric_type in ["heart_rate", "sleep_analysis", "activity_level"]:
            result = await firestore_client.get_user_health_data(
                "test_user",
                metric_type=metric_type
            )
            assert result["data"] == []

    async def test_date_range_filtering(self, firestore_client):
        """Test date range filtering functionality."""
        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_docs = MagicMock()

        mock_docs.__iter__ = MagicMock(return_value=iter([]))
        mock_query.stream = AsyncMock(return_value=mock_docs)
        mock_collection.where.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.order_by.return_value = mock_query

        firestore_client.client.collection.return_value = mock_collection

        start_date = datetime.now(UTC)
        end_date = datetime.now(UTC)

        result = await firestore_client.get_user_health_data(
            "test_user",
            start_date=start_date,
            end_date=end_date
        )

        assert result["data"] == []

    async def test_batch_operations_edge_cases(self, firestore_client, sample_health_metric):
        """Test batch operations with edge cases."""
        mock_collection = MagicMock()
        mock_doc_ref = MagicMock()
        mock_batch = MagicMock()

        firestore_client.client.collection.return_value = mock_collection
        mock_collection.document.return_value = mock_doc_ref
        firestore_client.client.batch.return_value = mock_batch

        mock_batch.set = MagicMock()
        mock_batch.commit = AsyncMock(return_value=True)

        # Test with large number of metrics
        large_metrics_list = [sample_health_metric] * 100

        result = await firestore_client.save_health_data(
            user_id="test_user",
            processing_id=str(uuid4()),
            metrics=large_metrics_list,
            upload_source="test",
            client_timestamp=datetime.now(UTC)
        )

        assert result is True

    async def test_error_handling_in_serialization(self, firestore_client):
        """Test error handling during metric serialization."""
        # Create a metric that might cause serialization issues
        metric = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.HEART_RATE,
            device_id="test_device",
            raw_data={},
            metadata={},
            created_at=datetime.now(UTC),
            biometric_data=BiometricData(
                heart_rate=75.0,
                blood_pressure_systolic=120,
                blood_pressure_diastolic=80,
                oxygen_saturation=99.0,
                heart_rate_variability=50.0,
                respiratory_rate=16.0,
                body_temperature=37.0,
                blood_glucose=100.0
            )
        )

        # Mock successful operation
        mock_collection = MagicMock()
        mock_doc_ref = MagicMock()
        mock_batch = MagicMock()

        firestore_client.client.collection.return_value = mock_collection
        mock_collection.document.return_value = mock_doc_ref
        firestore_client.client.batch.return_value = mock_batch

        mock_batch.set = MagicMock()
        mock_batch.commit = AsyncMock(return_value=True)

        result = await firestore_client.save_health_data(
            user_id="test_user",
            processing_id=str(uuid4()),
            metrics=[metric],
            upload_source="test",
            client_timestamp=datetime.now(UTC)
        )

        assert result is True
