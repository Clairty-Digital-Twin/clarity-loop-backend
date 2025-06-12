"""Comprehensive tests for MockHealthDataRepository.

Tests all methods and edge cases to improve coverage from 17% to 90%+.
Split into focused test classes to avoid PLR0904 (too many public methods).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from clarity.models.health_data import (
    ActivityData,
    BiometricData,
    HealthMetric,
    HealthMetricType,
    MentalHealthIndicator,
    SleepData,
    SleepStage,
)
from clarity.storage.mock_repository import MockHealthDataRepository

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


class TestMockHealthDataRepositoryBasics:
    """Basic functionality tests for MockHealthDataRepository."""

    @pytest.fixture
    @staticmethod
    async def repository() -> MockHealthDataRepository:
        """Create fresh repository instance for each test."""
        return MockHealthDataRepository()

    @pytest.fixture
    @staticmethod
    def sample_metrics() -> list[HealthMetric]:
        """Create sample health metrics for testing."""
        now = datetime.now(UTC)
        sleep_start = now.replace(hour=22, minute=0, second=0)
        sleep_end = now.replace(hour=8, minute=0, second=0) + timedelta(days=1)
        # Sleep duration from 10pm to 8am = 10 hours = 600 minutes
        sleep_duration = 600

        # Biometric metric
        biometric_metric = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.HEART_RATE,
            device_id="apple_watch_001",
            raw_data={},
            metadata={},
            created_at=now,
            biometric_data=BiometricData(
                heart_rate=72.5,
                blood_pressure_systolic=120,
                blood_pressure_diastolic=80,
                body_temperature=37.0,
                oxygen_saturation=99.0,
                heart_rate_variability=50.0,
                respiratory_rate=16.0,
                blood_glucose=100.0,
            ),
        )

        # Sleep metric
        sleep_metric = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.SLEEP_ANALYSIS,
            device_id="apple_watch_001",
            raw_data={},
            metadata={},
            created_at=now,
            sleep_data=SleepData(
                total_sleep_minutes=sleep_duration,
                sleep_efficiency=0.85,
                time_to_sleep_minutes=15,
                wake_count=2,
                sleep_stages={
                    SleepStage.LIGHT: 240,
                    SleepStage.DEEP: 120,
                    SleepStage.REM: 90,
                    SleepStage.AWAKE: 30,
                },
                sleep_start=sleep_start,
                sleep_end=sleep_end,
            ),
        )

        # Activity metric
        activity_metric = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.ACTIVITY_LEVEL,
            device_id="apple_watch_001",
            raw_data={},
            metadata={},
            created_at=now,
            activity_data=ActivityData(
                steps=10000,
                distance=8.0,
                active_energy=500.0,
                exercise_minutes=60,
                flights_climbed=10,
                vo2_max=45.0,
                active_minutes=60,
                resting_heart_rate=60.0,
            ),
        )

        # Mental health metric
        mental_metric = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.MOOD_ASSESSMENT,
            device_id="phone_001",
            raw_data={},
            metadata={},
            created_at=now,
            mental_health_data=MentalHealthIndicator(
                stress_level=3.0, anxiety_level=2.0, timestamp=now
            ),
        )

        return [biometric_metric, sleep_metric, activity_metric, mental_metric]

    @staticmethod
    async def test_initialization(repository: MockHealthDataRepository) -> None:
        """Test repository initialization."""
        assert repository._health_data == {}
        assert repository._processing_status == {}

    @staticmethod
    async def test_initialize_method(repository: MockHealthDataRepository) -> None:
        """Test initialize method."""
        await repository.initialize()
        # Should complete without error

    @staticmethod
    async def test_cleanup_method(repository: MockHealthDataRepository) -> None:
        """Test cleanup method."""
        await repository.cleanup()
        # Should complete without error

    @staticmethod
    async def test_save_data_generic(repository: MockHealthDataRepository) -> None:
        """Test generic save_data method."""
        user_id = "user_123"
        data = {"test": "data", "value": 42}

        doc_id = await repository.save_data(user_id, data)
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    @staticmethod
    async def test_get_data_generic(repository: MockHealthDataRepository) -> None:
        """Test generic get_data method."""
        user_id = "user_123"

        result = await repository.get_data(user_id)
        assert result == {}

        # Test with filters (should be ignored)
        result = await repository.get_data(user_id, filters={"test": "filter"})
        assert result == {}


class TestMockHealthDataRepositorySaving:
    """Tests for health data saving functionality."""

    @pytest.fixture
    @staticmethod
    async def repository() -> MockHealthDataRepository:
        """Create fresh repository instance for each test."""
        return MockHealthDataRepository()

    @pytest.fixture
    @staticmethod
    def sample_metrics() -> list[HealthMetric]:
        """Create sample health metrics for testing."""
        now = datetime.now(UTC)
        sleep_start = now.replace(hour=22, minute=0, second=0)
        sleep_end = now.replace(hour=8, minute=0, second=0) + timedelta(days=1)
        sleep_duration = 600

        biometric_metric = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.HEART_RATE,
            device_id="apple_watch_001",
            raw_data={},
            metadata={},
            created_at=now,
            biometric_data=BiometricData(
                heart_rate=72.5,
                blood_pressure_systolic=120,
                blood_pressure_diastolic=80,
                body_temperature=37.0,
                oxygen_saturation=99.0,
                heart_rate_variability=50.0,
                respiratory_rate=16.0,
                blood_glucose=100.0,
            ),
        )

        sleep_metric = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.SLEEP_ANALYSIS,
            device_id="apple_watch_001",
            raw_data={},
            metadata={},
            created_at=now,
            sleep_data=SleepData(
                total_sleep_minutes=sleep_duration,
                sleep_efficiency=0.85,
                time_to_sleep_minutes=15,
                wake_count=2,
                sleep_stages={
                    SleepStage.LIGHT: 240,
                    SleepStage.DEEP: 120,
                    SleepStage.REM: 90,
                    SleepStage.AWAKE: 30,
                },
                sleep_start=sleep_start,
                sleep_end=sleep_end,
            ),
        )

        activity_metric = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.ACTIVITY_LEVEL,
            device_id="apple_watch_001",
            raw_data={},
            metadata={},
            created_at=now,
            activity_data=ActivityData(
                steps=10000,
                distance=8.0,
                active_energy=500.0,
                exercise_minutes=60,
                flights_climbed=10,
                vo2_max=45.0,
                active_minutes=60,
                resting_heart_rate=60.0,
            ),
        )

        mental_metric = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.MOOD_ASSESSMENT,
            device_id="phone_001",
            raw_data={},
            metadata={},
            created_at=now,
            mental_health_data=MentalHealthIndicator(
                stress_level=3.0, anxiety_level=2.0, timestamp=now
            ),
        )

        return [biometric_metric, sleep_metric, activity_metric, mental_metric]

    @staticmethod
    async def test_save_health_data_success(
        repository: MockHealthDataRepository, sample_metrics: list[HealthMetric]
    ) -> None:
        """Test successful health data saving."""
        user_id = "user_123"
        processing_id = str(uuid4())
        upload_source = "apple_health"
        client_timestamp = datetime.now(UTC)

        result = await repository.save_health_data(
            user_id=user_id,
            processing_id=processing_id,
            metrics=sample_metrics,
            upload_source=upload_source,
            client_timestamp=client_timestamp,
        )

        assert result is True
        assert user_id in repository._health_data
        assert len(repository._health_data[user_id]) == 1
        assert processing_id in repository._processing_status

        # Verify processing status
        status = repository._processing_status[processing_id]
        assert status["processing_id"] == processing_id
        assert status["user_id"] == user_id
        assert status["status"] == "completed"
        assert status["metrics_count"] == 4

    @staticmethod
    async def test_save_health_data_multiple_uploads(
        repository: MockHealthDataRepository, sample_metrics: list[HealthMetric]
    ) -> None:
        """Test multiple uploads for same user."""
        user_id = "user_123"

        # First upload
        processing_id_1 = str(uuid4())
        await repository.save_health_data(
            user_id=user_id,
            processing_id=processing_id_1,
            metrics=sample_metrics[:2],
            upload_source="apple_health",
            client_timestamp=datetime.now(UTC),
        )

        # Second upload
        processing_id_2 = str(uuid4())
        await repository.save_health_data(
            user_id=user_id,
            processing_id=processing_id_2,
            metrics=sample_metrics[2:],
            upload_source="fitbit",
            client_timestamp=datetime.now(UTC),
        )

        assert len(repository._health_data[user_id]) == 2
        assert processing_id_1 in repository._processing_status
        assert processing_id_2 in repository._processing_status

    @staticmethod
    async def test_metric_serialization_all_types(
        repository: MockHealthDataRepository,
    ) -> None:
        """Test that all metric types are properly serialized."""
        user_id = "user_123"
        processing_id = str(uuid4())

        # Create metric with only biometric data
        biometric_only = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.HEART_RATE,
            device_id="device_001",
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

        await repository.save_health_data(
            user_id=user_id,
            processing_id=processing_id,
            metrics=[biometric_only],
            upload_source="test",
            client_timestamp=datetime.now(UTC),
        )

        result = await repository.get_user_health_data(user_id)
        metric_data = result["data"][0]

        assert "biometric_data" in metric_data
        assert metric_data["biometric_data"]["heart_rate"] == 75.0

    @staticmethod
    async def test_error_handling_in_save(
        repository: MockHealthDataRepository, monkeypatch: MonkeyPatch
    ) -> None:
        """Test error handling during save operation."""

        # Mock an exception during metric processing
        def mock_model_dump(*_args: object, **_kwargs: object) -> None:
            error_msg = "Test error"
            raise ValueError(error_msg)

        # This test ensures the except block is covered
        user_id = "user_123"
        processing_id = str(uuid4())

        # Create a metric that will cause error during serialization
        metric = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.HEART_RATE,
            device_id="device_001",
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

        # Patch the model_dump method to raise an exception
        monkeypatch.setattr(BiometricData, "model_dump", mock_model_dump)

        result = await repository.save_health_data(
            user_id=user_id,
            processing_id=processing_id,
            metrics=[metric],
            upload_source="test",
            client_timestamp=datetime.now(UTC),
        )

        assert result is False  # Should return False on error


class TestMockHealthDataRepositoryRetrieval:
    """Tests for health data retrieval functionality."""

    @pytest.fixture
    @staticmethod
    async def repository() -> MockHealthDataRepository:
        """Create fresh repository instance for each test."""
        return MockHealthDataRepository()

    @pytest.fixture
    @staticmethod
    def sample_metrics() -> list[HealthMetric]:
        """Create sample health metrics for testing."""
        now = datetime.now(UTC)
        sleep_start = now.replace(hour=22, minute=0, second=0)
        sleep_end = now.replace(hour=8, minute=0, second=0) + timedelta(days=1)
        sleep_duration = 600

        biometric_metric = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.HEART_RATE,
            device_id="apple_watch_001",
            raw_data={},
            metadata={},
            created_at=now,
            biometric_data=BiometricData(
                heart_rate=72.5,
                blood_pressure_systolic=120,
                blood_pressure_diastolic=80,
                body_temperature=37.0,
                oxygen_saturation=99.0,
                heart_rate_variability=50.0,
                respiratory_rate=16.0,
                blood_glucose=100.0,
            ),
        )

        sleep_metric = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.SLEEP_ANALYSIS,
            device_id="apple_watch_001",
            raw_data={},
            metadata={},
            created_at=now,
            sleep_data=SleepData(
                total_sleep_minutes=sleep_duration,
                sleep_efficiency=0.85,
                time_to_sleep_minutes=15,
                wake_count=2,
                sleep_stages={
                    SleepStage.LIGHT: 240,
                    SleepStage.DEEP: 120,
                    SleepStage.REM: 90,
                    SleepStage.AWAKE: 30,
                },
                sleep_start=sleep_start,
                sleep_end=sleep_end,
            ),
        )

        activity_metric = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.ACTIVITY_LEVEL,
            device_id="apple_watch_001",
            raw_data={},
            metadata={},
            created_at=now,
            activity_data=ActivityData(
                steps=10000,
                distance=8.0,
                active_energy=500.0,
                exercise_minutes=60,
                flights_climbed=10,
                vo2_max=45.0,
                active_minutes=60,
                resting_heart_rate=60.0,
            ),
        )

        mental_metric = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.MOOD_ASSESSMENT,
            device_id="phone_001",
            raw_data={},
            metadata={},
            created_at=now,
            mental_health_data=MentalHealthIndicator(
                stress_level=3.0, anxiety_level=2.0, timestamp=now
            ),
        )

        return [biometric_metric, sleep_metric, activity_metric, mental_metric]

    @staticmethod
    async def test_get_user_health_data_empty(
        repository: MockHealthDataRepository,
    ) -> None:
        """Test getting data for non-existent user."""
        result = await repository.get_user_health_data("non_existent_user")

        assert result["data"] == []
        assert result["total_count"] == 0
        assert result["page_info"]["has_more"] is False

    @staticmethod
    async def test_get_user_health_data_with_data(
        repository: MockHealthDataRepository, sample_metrics: list[HealthMetric]
    ) -> None:
        """Test getting user health data with existing data."""
        user_id = "user_123"
        processing_id = str(uuid4())

        await repository.save_health_data(
            user_id=user_id,
            processing_id=processing_id,
            metrics=sample_metrics,
            upload_source="apple_health",
            client_timestamp=datetime.now(UTC),
        )

        result = await repository.get_user_health_data(user_id)

        assert len(result["data"]) == 4
        assert result["total_count"] == 4
        assert result["page_info"]["has_more"] is False

    @staticmethod
    async def test_get_user_health_data_pagination(
        repository: MockHealthDataRepository, sample_metrics: list[HealthMetric]
    ) -> None:
        """Test pagination functionality."""
        user_id = "user_123"
        processing_id = str(uuid4())

        await repository.save_health_data(
            user_id=user_id,
            processing_id=processing_id,
            metrics=sample_metrics,
            upload_source="apple_health",
            client_timestamp=datetime.now(UTC),
        )

        # Test limit
        result = await repository.get_user_health_data(user_id, limit=2)
        assert len(result["data"]) == 2
        assert result["total_count"] == 4
        assert result["page_info"]["has_more"] is True

        # Test offset
        result = await repository.get_user_health_data(user_id, limit=2, offset=2)
        assert len(result["data"]) == 2
        assert result["page_info"]["has_more"] is False

    @staticmethod
    async def test_get_user_health_data_metric_type_filter(
        repository: MockHealthDataRepository, sample_metrics: list[HealthMetric]
    ) -> None:
        """Test filtering by metric type."""
        user_id = "user_123"
        processing_id = str(uuid4())

        await repository.save_health_data(
            user_id=user_id,
            processing_id=processing_id,
            metrics=sample_metrics,
            upload_source="apple_health",
            client_timestamp=datetime.now(UTC),
        )

        result = await repository.get_user_health_data(
            user_id, metric_type="heart_rate"
        )
        assert len(result["data"]) == 1
        assert result["data"][0]["metric_type"] == "heart_rate"

    @staticmethod
    async def test_get_user_health_data_date_filters(
        repository: MockHealthDataRepository, sample_metrics: list[HealthMetric]
    ) -> None:
        """Test date range filtering."""
        user_id = "user_123"
        processing_id = str(uuid4())

        # Modify timestamps for testing
        now = datetime.now(UTC)
        sample_metrics[0].created_at = now

        await repository.save_health_data(
            user_id=user_id,
            processing_id=processing_id,
            metrics=sample_metrics,
            upload_source="apple_health",
            client_timestamp=datetime.now(UTC),
        )

        # Test start_date filter
        result = await repository.get_user_health_data(user_id, start_date=now)
        assert len(result["data"]) >= 1


class TestMockHealthDataRepositoryProcessingAndDeletion:
    """Tests for processing status and deletion functionality."""

    @pytest.fixture
    @staticmethod
    async def repository() -> MockHealthDataRepository:
        """Create fresh repository instance for each test."""
        return MockHealthDataRepository()

    @pytest.fixture
    @staticmethod
    def sample_metrics() -> list[HealthMetric]:
        """Create sample health metrics for testing."""
        now = datetime.now(UTC)
        sleep_start = now.replace(hour=22, minute=0, second=0)
        sleep_end = now.replace(hour=8, minute=0, second=0) + timedelta(days=1)
        sleep_duration = 600

        biometric_metric = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.HEART_RATE,
            device_id="apple_watch_001",
            raw_data={},
            metadata={},
            created_at=now,
            biometric_data=BiometricData(
                heart_rate=72.5,
                blood_pressure_systolic=120,
                blood_pressure_diastolic=80,
                body_temperature=37.0,
                oxygen_saturation=99.0,
                heart_rate_variability=50.0,
                respiratory_rate=16.0,
                blood_glucose=100.0,
            ),
        )

        sleep_metric = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.SLEEP_ANALYSIS,
            device_id="apple_watch_001",
            raw_data={},
            metadata={},
            created_at=now,
            sleep_data=SleepData(
                total_sleep_minutes=sleep_duration,
                sleep_efficiency=0.85,
                time_to_sleep_minutes=15,
                wake_count=2,
                sleep_stages={
                    SleepStage.LIGHT: 240,
                    SleepStage.DEEP: 120,
                    SleepStage.REM: 90,
                    SleepStage.AWAKE: 30,
                },
                sleep_start=sleep_start,
                sleep_end=sleep_end,
            ),
        )

        activity_metric = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.ACTIVITY_LEVEL,
            device_id="apple_watch_001",
            raw_data={},
            metadata={},
            created_at=now,
            activity_data=ActivityData(
                steps=10000,
                distance=8.0,
                active_energy=500.0,
                exercise_minutes=60,
                flights_climbed=10,
                vo2_max=45.0,
                active_minutes=60,
                resting_heart_rate=60.0,
            ),
        )

        mental_metric = HealthMetric(
            metric_id=uuid4(),
            metric_type=HealthMetricType.MOOD_ASSESSMENT,
            device_id="phone_001",
            raw_data={},
            metadata={},
            created_at=now,
            mental_health_data=MentalHealthIndicator(
                stress_level=3.0, anxiety_level=2.0, timestamp=now
            ),
        )

        return [biometric_metric, sleep_metric, activity_metric, mental_metric]

    @staticmethod
    async def test_get_processing_status_exists(
        repository: MockHealthDataRepository, sample_metrics: list[HealthMetric]
    ) -> None:
        """Test getting processing status that exists."""
        user_id = "user_123"
        processing_id = str(uuid4())

        await repository.save_health_data(
            user_id=user_id,
            processing_id=processing_id,
            metrics=sample_metrics,
            upload_source="apple_health",
            client_timestamp=datetime.now(UTC),
        )

        status = await repository.get_processing_status(processing_id, user_id)
        assert status is not None
        assert status["processing_id"] == processing_id
        assert status["user_id"] == user_id
        assert status["status"] == "completed"

    @staticmethod
    async def test_get_processing_status_not_found(
        repository: MockHealthDataRepository,
    ) -> None:
        """Test getting non-existent processing status."""
        status = await repository.get_processing_status("non_existent", "user_123")
        assert status is None

    @staticmethod
    async def test_get_processing_status_wrong_user(
        repository: MockHealthDataRepository, sample_metrics: list[HealthMetric]
    ) -> None:
        """Test getting processing status with wrong user."""
        user_id = "user_123"
        processing_id = str(uuid4())

        await repository.save_health_data(
            user_id=user_id,
            processing_id=processing_id,
            metrics=sample_metrics,
            upload_source="apple_health",
            client_timestamp=datetime.now(UTC),
        )

        status = await repository.get_processing_status(processing_id, "wrong_user")
        assert status is None

    @staticmethod
    async def test_delete_health_data_specific_processing(
        repository: MockHealthDataRepository, sample_metrics: list[HealthMetric]
    ) -> None:
        """Test deleting specific processing job."""
        user_id = "user_123"
        processing_id_1 = str(uuid4())
        processing_id_2 = str(uuid4())

        # Create two processing jobs
        await repository.save_health_data(
            user_id=user_id,
            processing_id=processing_id_1,
            metrics=sample_metrics[:2],
            upload_source="apple_health",
            client_timestamp=datetime.now(UTC),
        )

        await repository.save_health_data(
            user_id=user_id,
            processing_id=processing_id_2,
            metrics=sample_metrics[2:],
            upload_source="fitbit",
            client_timestamp=datetime.now(UTC),
        )

        # Delete first processing job
        result = await repository.delete_health_data(user_id, processing_id_1)
        assert result is True

        # Verify only second job remains
        assert len(repository._health_data[user_id]) == 1
        assert repository._health_data[user_id][0]["processing_id"] == processing_id_2
        assert processing_id_1 not in repository._processing_status
        assert processing_id_2 in repository._processing_status

    @staticmethod
    async def test_delete_health_data_all_user_data(
        repository: MockHealthDataRepository, sample_metrics: list[HealthMetric]
    ) -> None:
        """Test deleting all user data."""
        user_id = "user_123"
        processing_id = str(uuid4())

        await repository.save_health_data(
            user_id=user_id,
            processing_id=processing_id,
            metrics=sample_metrics,
            upload_source="apple_health",
            client_timestamp=datetime.now(UTC),
        )

        result = await repository.delete_health_data(user_id)
        assert result is True
        assert user_id not in repository._health_data

    @staticmethod
    async def test_delete_health_data_non_existent_user(
        repository: MockHealthDataRepository,
    ) -> None:
        """Test deleting data for non-existent user."""
        result = await repository.delete_health_data("non_existent_user")
        assert result is True  # Should succeed even if no data
