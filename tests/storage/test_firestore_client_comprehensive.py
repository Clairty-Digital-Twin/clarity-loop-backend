"""Comprehensive tests for FirestoreHealthDataRepository.

Tests all methods and edge cases to improve coverage from 42% to 85%+.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from clarity.models.health_data import BiometricData, HealthMetric, HealthMetricType
from clarity.storage.firestore_client import FirestoreHealthDataRepository


class TestFirestoreHealthDataRepositoryComprehensive:
    """Comprehensive test coverage for FirestoreHealthDataRepository."""

    @pytest.fixture
    def mock_firestore_client(self):
        """Create a mock Firestore client."""
        mock_client = MagicMock()
        mock_client._get_db = AsyncMock()
        return mock_client

    @pytest.fixture
    def firestore_repository(self, mock_firestore_client):
        """Create FirestoreHealthDataRepository with mocked client."""
        with patch('clarity.storage.firestore_client.FirestoreClient') as mock_client_class:
            mock_client_class.return_value = mock_firestore_client
            repository = FirestoreHealthDataRepository(project_id="test-project")
            repository.client = mock_firestore_client
            return repository

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

    async def test_initialization_success(self):
        """Test successful FirestoreHealthDataRepository initialization."""
        with patch('clarity.storage.firestore_client.FirestoreClient'):
            repository = FirestoreHealthDataRepository(project_id="test-project")
            assert repository is not None

    async def test_initialization_failure(self):
        """Test FirestoreHealthDataRepository initialization failure."""
        with patch('clarity.storage.firestore_client.FirestoreClient') as mock_client:
            mock_client.side_effect = Exception("Connection failed")

            with pytest.raises(Exception):
                FirestoreHealthDataRepository(project_id="test-project")

    async def test_save_health_data_success(self, firestore_repository, sample_health_metric):
        """Test successful health data saving."""
        # Mock the store_health_data method
        firestore_repository.client.store_health_data = AsyncMock(return_value="doc_123")
        
        result = await firestore_repository.save_health_data(
            user_id="test_user",
            processing_id=str(uuid4()),
            metrics=[sample_health_metric],
            upload_source="test",
            client_timestamp=datetime.now(UTC)
        )

        assert result is True
        firestore_repository.client.store_health_data.assert_called_once()

    async def test_save_health_data_failure(self, firestore_repository, sample_health_metric):
        """Test health data saving failure."""
        # Mock the store_health_data method to fail
        firestore_repository.client.store_health_data = AsyncMock(side_effect=Exception("Firestore error"))

        result = await firestore_repository.save_health_data(
            user_id="test_user",
            processing_id=str(uuid4()),
            metrics=[sample_health_metric],
            upload_source="test",
            client_timestamp=datetime.now(UTC)
        )

        assert result is False

    async def test_save_health_data_empty_metrics(self, firestore_repository):
        """Test saving with empty metrics list."""
        firestore_repository.client.store_health_data = AsyncMock(return_value="doc_123")
        
        result = await firestore_repository.save_health_data(
            user_id="test_user",
            processing_id=str(uuid4()),
            metrics=[],
            upload_source="test",
            client_timestamp=datetime.now(UTC)
        )

        # Should still return True for empty metrics
        assert result is True

    async def test_get_user_health_data_success(self, firestore_repository):
        """Test successful user health data retrieval."""
        # Mock the query_documents method
        mock_data = [
            {
                "metric_id": "test_id",
                "metric_type": "heart_rate",
                "device_id": "test_device",
                "created_at": datetime.now(UTC).isoformat(),
                "biometric_data": {"heart_rate": 75.0}
            }
        ]
        
        firestore_repository.client.query_documents = AsyncMock(return_value=mock_data)
        firestore_repository.client.count_documents = AsyncMock(return_value=1)

        result = await firestore_repository.get_user_health_data("test_user")

        assert "data" in result
        assert "total_count" in result
        assert "page_info" in result
        assert result["total_count"] == 1

    async def test_get_user_health_data_with_filters(self, firestore_repository):
        """Test user health data retrieval with filters."""
        firestore_repository.client.query_documents = AsyncMock(return_value=[])
        firestore_repository.client.count_documents = AsyncMock(return_value=0)

        result = await firestore_repository.get_user_health_data(
            user_id="test_user",
            metric_type="heart_rate",
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC),
            limit=50,
            offset=10
        )

        assert result["data"] == []
        assert result["total_count"] == 0

    async def test_get_user_health_data_failure(self, firestore_repository):
        """Test user health data retrieval failure."""
        firestore_repository.client.query_documents = AsyncMock(side_effect=Exception("Firestore error"))

        result = await firestore_repository.get_user_health_data("test_user")

        # Should return empty result on failure
        assert result["data"] == []
        assert result["total_count"] == 0

    async def test_get_processing_status_exists(self, firestore_repository):
        """Test getting processing status that exists."""
        mock_status = {
            "processing_id": "test_id",
            "user_id": "test_user",
            "status": "completed"
        }
        
        firestore_repository.client.get_processing_status = AsyncMock(return_value=mock_status)

        result = await firestore_repository.get_processing_status("test_id", "test_user")

        assert result is not None
        assert result["processing_id"] == "test_id"

    async def test_get_processing_status_not_found(self, firestore_repository):
        """Test getting non-existent processing status."""
        firestore_repository.client.get_processing_status = AsyncMock(return_value=None)

        result = await firestore_repository.get_processing_status("non_existent", "test_user")

        assert result is None

    async def test_get_processing_status_wrong_user(self, firestore_repository):
        """Test getting processing status with wrong user."""
        firestore_repository.client.get_processing_status = AsyncMock(return_value=None)

        result = await firestore_repository.get_processing_status("test_id", "wrong_user")

        assert result is None

    async def test_get_processing_status_failure(self, firestore_repository):
        """Test processing status retrieval failure."""
        firestore_repository.client.get_processing_status = AsyncMock(side_effect=Exception("Firestore error"))

        result = await firestore_repository.get_processing_status("test_id", "test_user")

        assert result is None

    async def test_delete_health_data_specific_processing(self, firestore_repository):
        """Test deleting specific processing job."""
        firestore_repository.client.delete_documents = AsyncMock(return_value=1)

        result = await firestore_repository.delete_health_data("test_user", "test_processing_id")

        assert result is True
        firestore_repository.client.delete_documents.assert_called()

    async def test_delete_health_data_all_user_data(self, firestore_repository):
        """Test deleting all user data."""
        firestore_repository.client.delete_user_data = AsyncMock(return_value=5)

        result = await firestore_repository.delete_health_data("test_user")

        assert result is True
        firestore_repository.client.delete_user_data.assert_called_with("test_user")

    async def test_delete_health_data_failure(self, firestore_repository):
        """Test health data deletion failure."""
        firestore_repository.client.delete_user_data = AsyncMock(side_effect=Exception("Firestore error"))

        result = await firestore_repository.delete_health_data("test_user")

        assert result is False

    async def test_save_data_generic_success(self, firestore_repository):
        """Test generic save_data method success."""
        firestore_repository.client.create_document = AsyncMock(return_value="doc_123")

        result = await firestore_repository.save_data("test_user", {"test": "data"})

        assert result == "doc_123"

    async def test_save_data_generic_failure(self, firestore_repository):
        """Test generic save_data method failure."""
        firestore_repository.client.create_document = AsyncMock(side_effect=Exception("Firestore error"))

        with pytest.raises(Exception):
            await firestore_repository.save_data("test_user", {"test": "data"})

    async def test_get_data_generic_success(self, firestore_repository):
        """Test generic get_data method success."""
        mock_data = [{"test": "data"}]
        firestore_repository.client.query_documents = AsyncMock(return_value=mock_data)

        result = await firestore_repository.get_data("test_user")

        assert "user_id" in result
        assert "data" in result

    async def test_get_data_generic_with_filters(self, firestore_repository):
        """Test generic get_data method with filters."""
        mock_data = [{"test": "data"}]
        firestore_repository.client.query_documents = AsyncMock(return_value=mock_data)

        result = await firestore_repository.get_data("test_user", filters={"status": "active"})

        assert "user_id" in result
        assert "data" in result

    async def test_get_data_generic_failure(self, firestore_repository):
        """Test generic get_data method failure."""
        firestore_repository.client.query_documents = AsyncMock(side_effect=Exception("Firestore error"))

        result = await firestore_repository.get_data("test_user")

        # Should return empty data structure on failure
        assert result["data"] == []

    async def test_initialize_method(self, firestore_repository):
        """Test initialize method."""
        await firestore_repository.initialize()
        # Should complete without error

    async def test_cleanup_method(self, firestore_repository):
        """Test cleanup method."""
        firestore_repository.client.close = AsyncMock()
        
        await firestore_repository.cleanup()
        
        firestore_repository.client.close.assert_called_once()

    async def test_pagination_edge_cases(self, firestore_repository):
        """Test pagination edge cases."""
        firestore_repository.client.query_documents = AsyncMock(return_value=[])
        firestore_repository.client.count_documents = AsyncMock(return_value=0)

        # Test with very large offset
        result = await firestore_repository.get_user_health_data(
            "test_user", 
            limit=10, 
            offset=1000
        )

        assert result["data"] == []
        assert result["page_info"]["has_more"] is False

    async def test_metric_type_filtering(self, firestore_repository):
        """Test metric type filtering."""
        firestore_repository.client.query_documents = AsyncMock(return_value=[])
        firestore_repository.client.count_documents = AsyncMock(return_value=0)

        result = await firestore_repository.get_user_health_data(
            "test_user",
            metric_type="heart_rate"
        )

        # Verify the query was called with correct filters
        firestore_repository.client.query_documents.assert_called()
        args, kwargs = firestore_repository.client.query_documents.call_args
        
        # Should have filters for user_id and metric_type
        assert len(args) >= 2  # collection name and filters

    async def test_date_range_filtering(self, firestore_repository):
        """Test date range filtering."""
        firestore_repository.client.query_documents = AsyncMock(return_value=[])
        firestore_repository.client.count_documents = AsyncMock(return_value=0)

        start_date = datetime(2023, 1, 1, tzinfo=UTC)
        end_date = datetime(2023, 12, 31, tzinfo=UTC)

        result = await firestore_repository.get_user_health_data(
            "test_user",
            start_date=start_date,
            end_date=end_date
        )

        # Verify the query was called with date filters
        firestore_repository.client.query_documents.assert_called()
        args, kwargs = firestore_repository.client.query_documents.call_args
        
        # Should have filters for date range
        assert len(args) >= 2  # collection name and filters

    async def test_batch_operations_edge_cases(self, firestore_repository, sample_health_metric):
        """Test batch operations with edge cases."""
        firestore_repository.client.store_health_data = AsyncMock(return_value="doc_123")

        # Test with very large metrics list
        large_metrics_list = [sample_health_metric] * 100

        result = await firestore_repository.save_health_data(
            user_id="test_user",
            processing_id=str(uuid4()),
            metrics=large_metrics_list,
            upload_source="test",
            client_timestamp=datetime.now(UTC)
        )

        assert result is True

    async def test_error_handling_in_serialization(self, firestore_repository):
        """Test error handling during metric serialization."""
        # Create a problematic metric that might cause serialization issues
        problematic_metric = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.HEART_RATE,
            device_id="test_device",
            raw_data={"problematic": float('inf')},  # This might cause JSON serialization issues
            metadata={},
            created_at=datetime.now(UTC)
        )
        
        firestore_repository.client.store_health_data = AsyncMock(side_effect=Exception("Serialization error"))

        result = await firestore_repository.save_health_data(
            user_id="test_user",
            processing_id=str(uuid4()),
            metrics=[problematic_metric],
            upload_source="test",
            client_timestamp=datetime.now(UTC)
        )

        # Should handle the error gracefully
        assert result is False
