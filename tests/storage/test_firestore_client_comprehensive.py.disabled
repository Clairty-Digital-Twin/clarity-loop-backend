"""Comprehensive tests for FirestoreHealthDataRepository.

Tests all methods and edge cases to improve coverage from 42% to 85%+.
Split into focused test classes to avoid PLR0904 (too many public methods).
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from clarity.models.health_data import BiometricData, HealthMetric, HealthMetricType
from clarity.storage.firestore_client import (
    FirestoreError,
    FirestoreHealthDataRepository,
)


class TestFirestoreHealthDataRepositoryBasic:
    """Basic initialization and setup tests for FirestoreHealthDataRepository."""

    @pytest.fixture
    @staticmethod
    def mock_firestore_client() -> MagicMock:
        """Create a mock Firestore client."""
        mock_client = MagicMock()
        mock_client._get_db = AsyncMock()  # type: ignore[method-assign]
        return mock_client

    @pytest.fixture
    @staticmethod
    def firestore_repository(
        mock_firestore_client: MagicMock,
    ) -> FirestoreHealthDataRepository:
        """Create FirestoreHealthDataRepository with mocked client."""
        with patch(
            "clarity.storage.firestore_client.FirestoreClient"
        ) as mock_client_class:
            mock_client_class.return_value = mock_firestore_client
            repository = FirestoreHealthDataRepository(project_id="test-project")
            repository._firestore_client = mock_firestore_client
            return repository

    @pytest.fixture
    @staticmethod
    def sample_health_metric() -> HealthMetric:
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
                blood_glucose=100.0,
            ),
        )

    @staticmethod
    async def test_initialization_success() -> None:
        """Test successful FirestoreHealthDataRepository initialization."""
        with patch("clarity.storage.firestore_client.FirestoreClient"):
            repository = FirestoreHealthDataRepository(project_id="test-project")
            assert repository is not None

    @staticmethod
    async def test_initialization_failure() -> None:
        """Test FirestoreHealthDataRepository initialization failure."""
        with patch("clarity.storage.firestore_client.FirestoreClient") as mock_client:
            mock_client.side_effect = Exception("Connection failed")

            with pytest.raises(Exception, match="Connection failed"):
                FirestoreHealthDataRepository(project_id="test-project")

    @staticmethod
    async def test_initialize_method(
        firestore_repository: FirestoreHealthDataRepository,
    ) -> None:
        """Test initialize method."""
        firestore_repository._firestore_client.health_check = AsyncMock(return_value={"status": "healthy"})  # type: ignore[method-assign]
        await firestore_repository.initialize()
        firestore_repository._firestore_client.health_check.assert_called_once()

    @staticmethod
    async def test_cleanup_method(
        firestore_repository: FirestoreHealthDataRepository,
    ) -> None:
        """Test cleanup method."""
        firestore_repository._firestore_client.close = AsyncMock()  # type: ignore[method-assign]
        await firestore_repository.cleanup()
        firestore_repository._firestore_client.close.assert_called_once()


class TestFirestoreHealthDataSaving:
    """Tests for health data saving operations."""

    @pytest.fixture
    @staticmethod
    def mock_firestore_client() -> MagicMock:
        """Create a mock Firestore client."""
        mock_client = MagicMock()
        mock_client._get_db = AsyncMock()  # type: ignore[method-assign]
        return mock_client

    @pytest.fixture
    @staticmethod
    def firestore_repository(
        mock_firestore_client: MagicMock,
    ) -> FirestoreHealthDataRepository:
        """Create FirestoreHealthDataRepository with mocked client."""
        with patch(
            "clarity.storage.firestore_client.FirestoreClient"
        ) as mock_client_class:
            mock_client_class.return_value = mock_firestore_client
            repository = FirestoreHealthDataRepository(project_id="test-project")
            repository._firestore_client = mock_firestore_client
            return repository

    @pytest.fixture
    @staticmethod
    def sample_health_metric() -> HealthMetric:
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
                blood_glucose=100.0,
            ),
        )

    @staticmethod
    async def test_save_health_data_success(
        firestore_repository: FirestoreHealthDataRepository,
        sample_health_metric: HealthMetric,
    ) -> None:
        """Test successful health data saving."""
        firestore_repository._firestore_client.create_document = AsyncMock(return_value="doc_123")  # type: ignore[method-assign]
        result = await firestore_repository.save_health_data(
            user_id="test_user",
            processing_id=str(uuid4()),
            metrics=[sample_health_metric],
            upload_source="test",
            client_timestamp=datetime.now(UTC),
        )
        assert result is True
        firestore_repository._firestore_client.create_document.assert_called()

    @staticmethod
    async def test_save_health_data_failure(
        firestore_repository: FirestoreHealthDataRepository,
        sample_health_metric: HealthMetric,
    ) -> None:
        """Test health data saving failure."""
        firestore_repository._firestore_client.create_document = AsyncMock(side_effect=Exception("Firestore error"))  # type: ignore[method-assign]
        result = await firestore_repository.save_health_data(
            user_id="test_user",
            processing_id=str(uuid4()),
            metrics=[sample_health_metric],
            upload_source="test",
            client_timestamp=datetime.now(UTC),
        )
        assert result is False

    @staticmethod
    async def test_save_health_data_empty_metrics(
        firestore_repository: FirestoreHealthDataRepository,
    ) -> None:
        """Test saving with empty metrics list."""
        firestore_repository._firestore_client.create_document = AsyncMock(return_value="doc_123")  # type: ignore[method-assign]
        result = await firestore_repository.save_health_data(
            user_id="test_user",
            processing_id=str(uuid4()),
            metrics=[],
            upload_source="test",
            client_timestamp=datetime.now(UTC),
        )
        assert result is True

    @staticmethod
    async def test_save_data_generic_success(
        firestore_repository: FirestoreHealthDataRepository,
    ) -> None:
        """Test generic save_data method success."""
        firestore_repository._firestore_client.create_document = AsyncMock(return_value="doc_123")  # type: ignore[method-assign]
        result = await firestore_repository.save_data("test_user", {"test": "data"})
        assert result == "doc_123"

    @staticmethod
    async def test_save_data_generic_failure(
        firestore_repository: FirestoreHealthDataRepository,
    ) -> None:
        """Test generic save_data method failure."""
        firestore_repository._firestore_client.create_document = AsyncMock(side_effect=Exception("Firestore error"))  # type: ignore[method-assign]
        with pytest.raises(Exception, match="Firestore error"):
            await firestore_repository.save_data("test_user", {"test": "data"})

    @staticmethod
    async def test_batch_operations_edge_cases(
        firestore_repository: FirestoreHealthDataRepository,
        sample_health_metric: HealthMetric,
    ) -> None:
        """Test batch operations with edge cases."""
        firestore_repository._firestore_client.create_document = AsyncMock(return_value="doc_123")  # type: ignore[method-assign]
        large_metrics_list = [sample_health_metric] * 100
        result = await firestore_repository.save_health_data(
            user_id="test_user",
            processing_id=str(uuid4()),
            metrics=large_metrics_list,
            upload_source="test",
            client_timestamp=datetime.now(UTC),
        )
        assert result is True

    @staticmethod
    async def test_error_handling_in_serialization(
        firestore_repository: FirestoreHealthDataRepository,
    ) -> None:
        """Test error handling during metric serialization."""
        problematic_metric = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.HEART_RATE,
            device_id="test_device",
            raw_data={"problematic": float("inf")},
            metadata={},
            created_at=datetime.now(UTC),
            biometric_data=BiometricData(
                heart_rate=75.0,
                blood_pressure_systolic=120,
                blood_pressure_diastolic=80,
                oxygen_saturation=None,
                heart_rate_variability=None,
                respiratory_rate=None,
                body_temperature=None,
                blood_glucose=None,
            ),
        )
        firestore_repository._firestore_client.create_document = AsyncMock(side_effect=Exception("Serialization error"))  # type: ignore[method-assign]
        result = await firestore_repository.save_health_data(
            user_id="test_user",
            processing_id=str(uuid4()),
            metrics=[problematic_metric],
            upload_source="test",
            client_timestamp=datetime.now(UTC),
        )
        assert result is False


class TestFirestoreHealthDataRetrieval:
    """Tests for health data retrieval operations."""

    @pytest.fixture
    @staticmethod
    def mock_firestore_client() -> MagicMock:
        mock_client = MagicMock()
        mock_client._get_db = AsyncMock()  # type: ignore[method-assign]
        return mock_client

    @pytest.fixture
    @staticmethod
    def firestore_repository(
        mock_firestore_client: MagicMock,
    ) -> FirestoreHealthDataRepository:
        with patch(
            "clarity.storage.firestore_client.FirestoreClient"
        ) as mock_client_class:
            mock_client_class.return_value = mock_firestore_client
            repository = FirestoreHealthDataRepository(project_id="test-project")
            repository._firestore_client = mock_firestore_client
            return repository

    @staticmethod
    async def test_get_user_health_data_success(
        firestore_repository: FirestoreHealthDataRepository,
    ) -> None:
        """Test successful user health data retrieval."""
        mock_data = [
            {
                "metric_id": "test_id",
                "metric_type": "heart_rate",
                "device_id": "test_device",
                "created_at": datetime.now(UTC).isoformat(),
                "biometric_data": {"heart_rate": 75.0},
            }
        ]
        firestore_repository._firestore_client.query_documents = AsyncMock(return_value=mock_data)  # type: ignore[method-assign]
        firestore_repository._firestore_client.count_documents = AsyncMock(return_value=1)  # type: ignore[method-assign]
        result = await firestore_repository.get_user_health_data("test_user")
        assert "metrics" in result
        assert "pagination" in result
        assert result["pagination"]["total"] == 1

    @staticmethod
    async def test_get_user_health_data_with_filters(
        firestore_repository: FirestoreHealthDataRepository,
    ) -> None:
        """Test user health data retrieval with filters."""
        firestore_repository._firestore_client.query_documents = AsyncMock(return_value=[])  # type: ignore[method-assign]
        firestore_repository._firestore_client.count_documents = AsyncMock(return_value=0)  # type: ignore[method-assign]
        result = await firestore_repository.get_user_health_data(
            user_id="test_user",
            metric_type="heart_rate",
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC),
            limit=50,
            offset=10,
        )
        assert result["metrics"] == []
        assert result["pagination"]["total"] == 0

    @staticmethod
    async def test_get_user_health_data_failure(
        firestore_repository: FirestoreHealthDataRepository,
    ) -> None:
        """Test user health data retrieval failure."""
        firestore_repository._firestore_client.query_documents = AsyncMock(side_effect=Exception("Firestore error"))  # type: ignore[method-assign]
        with pytest.raises(FirestoreError):
            await firestore_repository.get_user_health_data("test_user")

    @staticmethod
    async def test_get_data_generic_success(
        firestore_repository: FirestoreHealthDataRepository,
    ) -> None:
        """Test generic get_data method success."""
        mock_data = [{"test": "data"}]
        firestore_repository._firestore_client.query_documents = AsyncMock(return_value=mock_data)  # type: ignore[method-assign]
        result = await firestore_repository.get_data("test_user")
        assert "user_id" in result
        assert "data" in result

    @staticmethod
    async def test_get_data_generic_with_filters(
        firestore_repository: FirestoreHealthDataRepository,
    ) -> None:
        """Test generic get_data method with filters."""
        mock_data = [{"test": "data"}]
        firestore_repository._firestore_client.query_documents = AsyncMock(return_value=mock_data)  # type: ignore[method-assign]
        result = await firestore_repository.get_data(
            "test_user", filters={"status": "active"}
        )
        assert "user_id" in result
        assert "data" in result

    @staticmethod
    async def test_get_data_generic_failure(
        firestore_repository: FirestoreHealthDataRepository,
    ) -> None:
        """Test generic get_data method failure."""
        firestore_repository._firestore_client.query_documents = AsyncMock(side_effect=Exception("Firestore error"))  # type: ignore[method-assign]
        with pytest.raises(FirestoreError):
            await firestore_repository.get_data("test_user")

    @staticmethod
    async def test_pagination_edge_cases(
        firestore_repository: FirestoreHealthDataRepository,
    ) -> None:
        """Test pagination edge cases."""
        firestore_repository._firestore_client.query_documents = AsyncMock(return_value=[])  # type: ignore[method-assign]
        firestore_repository._firestore_client.count_documents = AsyncMock(return_value=0)  # type: ignore[method-assign]
        result = await firestore_repository.get_user_health_data(
            "test_user", limit=10, offset=1000
        )
        assert result["metrics"] == []
        assert result["pagination"]["has_more"] is False

    @staticmethod
    async def test_metric_type_filtering(
        firestore_repository: FirestoreHealthDataRepository,
    ) -> None:
        """Test metric type filtering."""
        firestore_repository._firestore_client.query_documents = AsyncMock(return_value=[])  # type: ignore[method-assign]
        firestore_repository._firestore_client.count_documents = AsyncMock(return_value=0)  # type: ignore[method-assign]
        await firestore_repository.get_user_health_data(
            "test_user", metric_type="heart_rate"
        )
        firestore_repository._firestore_client.query_documents.assert_called()
        call_args = firestore_repository._firestore_client.query_documents.call_args
        assert call_args is not None
        assert firestore_repository._firestore_client.query_documents.called

    @staticmethod
    async def test_date_range_filtering(
        firestore_repository: FirestoreHealthDataRepository,
    ) -> None:
        """Test date range filtering."""
        firestore_repository._firestore_client.query_documents = AsyncMock(return_value=[])  # type: ignore[method-assign]
        firestore_repository._firestore_client.count_documents = AsyncMock(return_value=0)  # type: ignore[method-assign]
        start_date = datetime(2023, 1, 1, tzinfo=UTC)
        end_date = datetime(2023, 12, 31, tzinfo=UTC)
        await firestore_repository.get_user_health_data(
            "test_user", start_date=start_date, end_date=end_date
        )
        firestore_repository._firestore_client.query_documents.assert_called()
        call_args = firestore_repository._firestore_client.query_documents.call_args
        assert call_args is not None
        assert firestore_repository._firestore_client.query_documents.called


class TestFirestoreStatusAndDeletion:
    """Tests for processing status and data deletion operations."""

    @pytest.fixture
    @staticmethod
    def mock_firestore_client() -> MagicMock:
        mock_client = MagicMock()
        mock_client._get_db = AsyncMock()  # type: ignore[method-assign]
        return mock_client

    @pytest.fixture
    @staticmethod
    def firestore_repository(
        mock_firestore_client: MagicMock,
    ) -> FirestoreHealthDataRepository:
        with patch(
            "clarity.storage.firestore_client.FirestoreClient"
        ) as mock_client_class:
            mock_client_class.return_value = mock_firestore_client
            repository = FirestoreHealthDataRepository(project_id="test-project")
            repository._firestore_client = mock_firestore_client
            return repository

    @staticmethod
    async def test_get_processing_status_exists(
        firestore_repository: FirestoreHealthDataRepository,
    ) -> None:
        """Test getting processing status that exists."""
        mock_status = {
            "processing_id": "test_id",
            "user_id": "test_user",
            "status": "completed",
        }
        firestore_repository._firestore_client.get_document = AsyncMock(return_value=mock_status)  # type: ignore[method-assign]
        result = await firestore_repository.get_processing_status(
            "test_id", "test_user"
        )
        assert result is not None
        assert result["processing_id"] == "test_id"

    @staticmethod
    async def test_get_processing_status_not_found(
        firestore_repository: FirestoreHealthDataRepository,
    ) -> None:
        """Test getting non-existent processing status."""
        firestore_repository._firestore_client.get_document = AsyncMock(return_value=None)  # type: ignore[method-assign]
        result = await firestore_repository.get_processing_status(
            "non_existent", "test_user"
        )
        assert result is None

    @staticmethod
    async def test_get_processing_status_wrong_user(
        firestore_repository: FirestoreHealthDataRepository,
    ) -> None:
        """Test getting processing status with wrong user."""
        firestore_repository._firestore_client.get_document = AsyncMock(return_value=None)  # type: ignore[method-assign]
        result = await firestore_repository.get_processing_status(
            "test_id", "wrong_user"
        )
        assert result is None

    @staticmethod
    async def test_get_processing_status_failure(
        firestore_repository: FirestoreHealthDataRepository,
    ) -> None:
        """Test processing status retrieval failure."""
        firestore_repository._firestore_client.get_document = AsyncMock(side_effect=Exception("Firestore error"))  # type: ignore[method-assign]
        result = await firestore_repository.get_processing_status(
            "test_id", "test_user"
        )
        assert result is None

    @staticmethod
    async def test_delete_health_data_specific_processing(
        firestore_repository: FirestoreHealthDataRepository,
    ) -> None:
        """Test deleting specific processing job."""
        firestore_repository._firestore_client.delete_documents = AsyncMock(return_value=1)  # type: ignore[method-assign]
        firestore_repository._firestore_client.delete_document = AsyncMock(return_value=True)  # type: ignore[method-assign]
        firestore_repository._firestore_client.create_document = AsyncMock(return_value="audit_123")  # type: ignore[method-assign]
        result = await firestore_repository.delete_health_data(
            "test_user", "test_processing_id"
        )
        assert result is True
        firestore_repository._firestore_client.delete_documents.assert_called()

    async def test_delete_health_data_all_user_data(  # noqa: PLR6301
        self, firestore_repository: FirestoreHealthDataRepository
    ) -> None:
        """Test deleting all user data."""
        firestore_repository._firestore_client.delete_documents = AsyncMock(return_value=5)  # type: ignore[method-assign]
        firestore_repository._firestore_client.create_document = AsyncMock(return_value="audit_123")  # type: ignore[method-assign]
        result = await firestore_repository.delete_health_data("test_user")
        assert result is True
        firestore_repository._firestore_client.delete_documents.assert_called()

    async def test_delete_health_data_failure(  # noqa: PLR6301
        self, firestore_repository: FirestoreHealthDataRepository
    ) -> None:
        """Test health data deletion failure."""
        firestore_repository._firestore_client.delete_documents = AsyncMock(side_effect=Exception("Firestore error"))  # type: ignore[method-assign]
        firestore_repository._firestore_client.create_document = AsyncMock(return_value="audit_123")  # type: ignore[method-assign]
        result = await firestore_repository.delete_health_data("test_user")
        assert result is False
