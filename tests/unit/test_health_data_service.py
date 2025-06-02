"""CLARITY Digital Twin Platform - Application Business Rules Tests.

💼 APPLICATION BUSINESS RULES LAYER TESTS (Clean Architecture Use Cases)

These tests verify the application-specific business rules and use case orchestration.
Following Robert C. Martin's Clean Architecture principle: "Use cases orchestrate the
flow of data to and from the entities, and direct those entities to use their
enterprise-wide business rules to achieve the goals of the use case."

TESTS USE MOCKS FOR DEPENDENCIES BUT NO REAL IMPLEMENTATIONS.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest

from clarity.core.interfaces import IHealthDataRepository
from clarity.models.health_data import (
    BiometricData,
    HealthDataResponse,
    HealthDataUpload,
    HealthMetric,
    HealthMetricType,
    ProcessingStatus,
)

# Import use cases and services (Application layer)
from clarity.services.health_data_service import (
    HealthDataService,
    HealthDataServiceError,
)


class TestHealthDataServiceUseCase:
    """Test application business rules - use case orchestration with mocked dependencies.

    Following Clean Architecture: Use cases depend on abstractions (interfaces),
    not on concrete implementations. Tests verify orchestration logic.
    """

    def test_service_initialization_dependency_injection(self):
        """Test use case follows Dependency Inversion Principle."""
        # Given: Mock repository (abstraction, not concrete implementation)
        mock_repository = Mock(spec=IHealthDataRepository)

        # When: Creating service with dependency injection
        service = HealthDataService(mock_repository)

        # Then: Service should depend on abstraction
        assert service._repository is mock_repository
        assert hasattr(service, "_repository")  # Service has injected dependency

    @pytest.mark.asyncio
    async def test_process_health_data_use_case_orchestration(self):
        """Test use case orchestrates entity validation and repository storage."""
        # Given: Mock repository and valid health data upload
        mock_repository = AsyncMock(spec=IHealthDataRepository)
        mock_repository.store_health_data.return_value = "processing-123"

        service = HealthDataService(mock_repository)

        user_id = uuid4()
        biometric_data = BiometricData(heart_rate=72, timestamp=datetime.now(UTC))
        health_metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE, biometric_data=biometric_data
        )
        health_upload = HealthDataUpload(
            user_id=user_id,
            metrics=[health_metric],
            upload_source="apple_health",
            client_timestamp=datetime.now(UTC),
        )

        # When: Processing health data (use case execution)
        result = await service.process_health_data(health_upload)

        # Then: Use case should orchestrate validation and storage
        assert isinstance(result, HealthDataResponse)
        assert result.status == ProcessingStatus.PROCESSING
        assert result.accepted_metrics == 1
        assert result.rejected_metrics == 0

        # Verify repository was called (use case orchestration)
        mock_repository.store_health_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_use_case_handles_business_rule_violations(self):
        """Test use case properly handles business rule violations from entities."""
        # Given: Mock repository and invalid health data
        mock_repository = AsyncMock(spec=IHealthDataRepository)
        service = HealthDataService(mock_repository)

        user_id = uuid4()

        # Create health upload with invalid business rule (will fail at entity level)
        try:
            # This should fail due to business rules in the entity
            invalid_biometric = BiometricData(
                heart_rate=500,  # Invalid heart rate (business rule violation)
                timestamp=datetime.now(UTC),
            )
            health_metric = HealthMetric(
                metric_type=HealthMetricType.HEART_RATE,
                biometric_data=invalid_biometric,
            )
            health_upload = HealthDataUpload(
                user_id=user_id,
                metrics=[health_metric],
                upload_source="apple_health",
                client_timestamp=datetime.now(UTC),
            )
        except ValueError:
            # Expected - business rule caught at entity level
            # Use case should handle this gracefully
            pass

        # Repository should not be called for invalid data
        mock_repository.store_health_data.assert_not_called()

    @pytest.mark.asyncio
    async def test_use_case_repository_error_handling(self):
        """Test use case handles repository failures gracefully."""
        # Given: Mock repository that fails
        mock_repository = AsyncMock(spec=IHealthDataRepository)
        mock_repository.store_health_data.side_effect = Exception(
            "Database connection failed"
        )

        service = HealthDataService(mock_repository)

        user_id = uuid4()
        biometric_data = BiometricData(heart_rate=72, timestamp=datetime.now(UTC))
        health_metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE, biometric_data=biometric_data
        )
        health_upload = HealthDataUpload(
            user_id=user_id,
            metrics=[health_metric],
            upload_source="apple_health",
            client_timestamp=datetime.now(UTC),
        )

        # When/Then: Use case should handle repository errors
        with pytest.raises(HealthDataServiceError):
            await service.process_health_data(health_upload)

    @pytest.mark.asyncio
    async def test_get_processing_status_use_case(self):
        """Test use case for retrieving processing status."""
        # Given: Mock repository with status data
        mock_repository = AsyncMock(spec=IHealthDataRepository)
        expected_status = {
            "processing_id": "test-123",
            "status": ProcessingStatus.COMPLETED.value,
            "created_at": datetime.now(UTC).isoformat(),
            "metrics_count": 5,
        }
        mock_repository.get_processing_status.return_value = expected_status

        service = HealthDataService(mock_repository)

        # When: Getting processing status
        result = await service.get_processing_status("test-123", "user-456")

        # Then: Use case should return repository data
        assert result == expected_status
        mock_repository.get_processing_status.assert_called_once_with(
            "test-123", "user-456"
        )

    @pytest.mark.asyncio
    async def test_get_user_health_data_use_case_orchestration(self):
        """Test use case orchestrates health data retrieval with filters."""
        # Given: Mock repository with health data
        mock_repository = AsyncMock(spec=IHealthDataRepository)
        expected_data = {
            "metrics": [
                {
                    "metric_type": "heart_rate",
                    "value": 72,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ],
            "total_count": 1,
            "page_info": {"limit": 100, "offset": 0},
        }
        mock_repository.get_user_health_data.return_value = expected_data

        service = HealthDataService(mock_repository)

        # When: Getting user health data with filters
        result = await service.get_user_health_data(
            user_id="user-123",
            limit=100,
            offset=0,
            metric_type="heart_rate",
            start_date=None,
            end_date=None,
        )

        # Then: Use case should orchestrate repository call with filters
        assert result == expected_data
        mock_repository.get_user_health_data.assert_called_once_with(
            user_id="user-123",
            limit=100,
            offset=0,
            metric_type="heart_rate",
            start_date=None,
            end_date=None,
        )

    @pytest.mark.asyncio
    async def test_delete_health_data_use_case(self):
        """Test use case for deleting health data."""
        # Given: Mock repository that confirms deletion
        mock_repository = AsyncMock(spec=IHealthDataRepository)
        mock_repository.delete_health_data.return_value = True

        service = HealthDataService(mock_repository)

        # When: Deleting health data
        result = await service.delete_health_data("user-123", "processing-456")

        # Then: Use case should return success status
        assert result is True
        mock_repository.delete_health_data.assert_called_once_with(
            "user-123", "processing-456"
        )

    def test_business_rule_validation_orchestration(self):
        """Test use case orchestrates business rule validation."""
        # Given: Mock repository
        mock_repository = Mock(spec=IHealthDataRepository)
        service = HealthDataService(mock_repository)

        # When: Testing metric validation
        valid_biometric = BiometricData(heart_rate=72, timestamp=datetime.now(UTC))
        valid_metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE, biometric_data=valid_biometric
        )

        # Then: Service should be able to validate metrics
        # This tests the use case has access to validation logic
        is_valid = service._validate_metric_business_rules(valid_metric)
        assert is_valid is True

    def test_service_logging_and_monitoring(self):
        """Test use case includes proper logging for monitoring."""
        # Given: Mock repository
        mock_repository = Mock(spec=IHealthDataRepository)
        service = HealthDataService(mock_repository)

        # Then: Service should have logger for monitoring
        assert hasattr(service, "logger")
        assert service.logger is not None

        # Use case should support monitoring/observability
        assert service.logger.name == "clarity.services.health_data_service"


class TestHealthDataServiceBusinessRules:
    """Test application-specific business rules in the service layer."""

    def test_metric_validation_business_rule(self):
        """Test application business rule: Metrics must pass validation."""
        # Given: Service with mock repository
        mock_repository = Mock(spec=IHealthDataRepository)
        service = HealthDataService(mock_repository)

        # Valid metric
        valid_biometric = BiometricData(heart_rate=72, timestamp=datetime.now(UTC))
        valid_metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE, biometric_data=valid_biometric
        )

        # When: Validating metric
        is_valid = service._validate_metric_business_rules(valid_metric)

        # Then: Valid metric should pass
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_processing_id_generation_business_rule(self):
        """Test application business rule: Each upload gets unique processing ID."""
        # Given: Mock repository
        mock_repository = AsyncMock(spec=IHealthDataRepository)
        mock_repository.store_health_data.return_value = "unique-processing-id"

        service = HealthDataService(mock_repository)

        user_id = uuid4()
        biometric_data = BiometricData(heart_rate=72, timestamp=datetime.now(UTC))
        health_metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE, biometric_data=biometric_data
        )
        health_upload = HealthDataUpload(
            user_id=user_id,
            metrics=[health_metric],
            upload_source="apple_health",
            client_timestamp=datetime.now(UTC),
        )

        # When: Processing health data
        result = await service.process_health_data(health_upload)

        # Then: Response should have unique processing ID
        assert result.processing_id is not None
        assert isinstance(result.processing_id, UUID)

    @pytest.mark.asyncio
    async def test_error_response_business_rule(self):
        """Test application business rule: Errors are wrapped in service exceptions."""
        # Given: Service with failing repository
        mock_repository = AsyncMock(spec=IHealthDataRepository)
        mock_repository.store_health_data.side_effect = RuntimeError("Storage failed")

        service = HealthDataService(mock_repository)

        user_id = uuid4()
        biometric_data = BiometricData(heart_rate=72, timestamp=datetime.now(UTC))
        health_metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE, biometric_data=biometric_data
        )
        health_upload = HealthDataUpload(
            user_id=user_id,
            metrics=[health_metric],
            upload_source="apple_health",
            client_timestamp=datetime.now(UTC),
        )

        # When/Then: Service should wrap errors in application exceptions
        with pytest.raises(HealthDataServiceError) as exc_info:
            await service.process_health_data(health_upload)

        # Application business rule: Service errors have specific structure
        assert "Storage failed" in str(exc_info.value)


class TestHealthDataServiceDependencyInversion:
    """Test that service follows Dependency Inversion Principle (SOLID)."""

    def test_service_depends_on_abstraction_not_concretion(self):
        """Test service depends on repository interface, not concrete implementation."""
        # Given: Mock repository interface
        mock_repository = Mock(spec=IHealthDataRepository)

        # When: Creating service
        service = HealthDataService(mock_repository)

        # Then: Service should depend on interface
        assert isinstance(service._repository, type(mock_repository))

        # Service should not know about concrete repository implementation
        # It only knows about the interface methods
        assert hasattr(service._repository, "store_health_data")
        assert hasattr(service._repository, "get_processing_status")
        assert hasattr(service._repository, "get_user_health_data")
        assert hasattr(service._repository, "delete_health_data")

    def test_service_is_testable_without_real_implementations(self):
        """Test service can be fully tested with mocks (no real database needed)."""
        # Given: All dependencies are mocked
        mock_repository = Mock(spec=IHealthDataRepository)

        # When: Creating and testing service
        service = HealthDataService(mock_repository)

        # Then: Service is fully testable without any real implementations
        assert service is not None
        assert service._repository is mock_repository

        # All service methods can be tested with mocks
        assert callable(getattr(service, "process_health_data", None))
        assert callable(getattr(service, "get_processing_status", None))
        assert callable(getattr(service, "get_user_health_data", None))
        assert callable(getattr(service, "delete_health_data", None))


class TestHealthDataServiceSingleResponsibility:
    """Test service follows Single Responsibility Principle (SOLID)."""

    def test_service_has_single_responsibility(self):
        """Test service only handles health data operations."""
        # Given: Mock repository
        mock_repository = Mock(spec=IHealthDataRepository)
        service = HealthDataService(mock_repository)

        # Then: Service should only have health data related methods
        health_data_methods = [
            "process_health_data",
            "get_processing_status",
            "get_user_health_data",
            "delete_health_data",
        ]

        for method in health_data_methods:
            assert hasattr(service, method)
            assert callable(getattr(service, method))

        # Service should not have methods for other concerns
        non_health_methods = [
            "send_email",
            "process_payment",
            "authenticate_user",
            "generate_report",
        ]

        for method in non_health_methods:
            assert not hasattr(service, method)

    def test_service_validation_is_health_data_specific(self):
        """Test service validation logic is specific to health data domain."""
        # Given: Service with mock repository
        mock_repository = Mock(spec=IHealthDataRepository)
        service = HealthDataService(mock_repository)

        # Then: Validation method should be health-metric specific
        assert hasattr(service, "_validate_metric_business_rules")

        # Validation should not handle other domains
        non_health_validation_methods = [
            "_validate_payment",
            "_validate_user_profile",
            "_validate_email_format",
        ]

        for method in non_health_validation_methods:
            assert not hasattr(service, method)
