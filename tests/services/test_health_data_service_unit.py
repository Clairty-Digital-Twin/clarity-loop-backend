"""Fast unit tests for health data service."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock
import uuid

import pytest

from clarity.models.health_data import (
    BiometricData,
    HealthDataUpload,
    HealthDataResponse,
    HealthMetric,
    HealthMetricType,
)
from clarity.services.health_data_service import (
    HealthDataService,
    HealthDataServiceError,
)


class TestHealthDataServiceHappyPath:
    """Test health data service happy path scenarios."""

    @pytest.mark.asyncio
    async def test_process_health_data_success(self):
        """Test successful health data processing."""
        # Create mock repository
        mock_repository = Mock()
        mock_repository.save_health_data = AsyncMock(return_value=True)

        # Create mock config provider
        mock_config = Mock()
        mock_config.get_config = Mock(return_value={})

        # Create service
        service = HealthDataService(mock_repository, mock_config)

        # Create test data
        processing_id = str(uuid.uuid4())
        upload_data = HealthDataUpload(
            user_id=str(uuid.uuid4()),
            upload_source="test",
            metrics=[
                HealthMetric(
                    metric_type=HealthMetricType.HEART_RATE,
                    biometric_data=BiometricData(heart_rate=72.0),
                )
            ],
            client_timestamp=datetime.now(UTC),
        )

        # Process data
        result = await service.process_health_data(upload_data)

        # Verify result
        assert isinstance(result, HealthDataResponse)
        assert result.processing_id is not None
        assert result.accepted_metrics == 1

        # Verify repository was called
        mock_repository.save_health_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_processing_status_found(self):
        """Test getting processing status when job exists."""
        # Create mock repository
        mock_repository = Mock()
        mock_repository.get_processing_status = AsyncMock(
            return_value={
                "processing_id": "test-id",
                "status": "completed",
                "created_at": datetime.now(UTC),
            }
        )

        # Create service
        service = HealthDataService(mock_repository, Mock())

        # Get status
        result = await service.get_processing_status("test-user", "test-id")

        # Verify result
        assert result is not None
        assert result["processing_id"] == "test-id"
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_delete_health_data_success(self):
        """Test successful health data deletion."""
        # Create mock repository
        mock_repository = Mock()
        mock_repository.delete_health_data = AsyncMock(return_value=True)

        # Create service
        service = HealthDataService(mock_repository, Mock())

        # Delete data
        result = await service.delete_health_data("test-user", "test-id")

        # Verify result
        assert result is True
        mock_repository.delete_health_data.assert_called_once_with(
            user_id="test-user", processing_id="test-id"
        )

    @pytest.mark.asyncio
    async def test_list_health_data_with_filters(self):
        """Test listing health data with filters."""
        # Create mock repository
        mock_repository = Mock()
        mock_repository.get_user_health_data = AsyncMock(
            return_value={
                "metrics": [{"id": "1"}, {"id": "2"}],
                "total": 2,
                "page": 1,
                "page_size": 10,
            }
        )

        # Create service
        service = HealthDataService(mock_repository, Mock())

        # List data with filters
        result = await service.get_user_health_data(
            user_id="test-user",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            limit=10,
            offset=0,
        )

        # Verify result
        assert len(result["metrics"]) == 2
        assert result["total"] == 2

        # Verify repository was called with correct parameters
        mock_repository.get_user_health_data.assert_called_once()
        call_kwargs = mock_repository.get_user_health_data.call_args[1]
        assert call_kwargs["user_id"] == "test-user"
        assert "start_date" in call_kwargs
        assert "end_date" in call_kwargs
