"""Tests for CLARITY service health checks."""

from __future__ import annotations

import asyncio  # noqa: F401 - needed for pytest.mark.asyncio
import time
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError, NoCredentialsError
import pytest

from clarity.startup.health_checks import (
    CircuitBreakerState,
    HealthCheckResult,
    ServiceHealthChecker,
    ServiceStatus,
)


class TestCircuitBreakerState:
    """Test circuit breaker functionality."""

    def test_initial_state(self) -> None:
        """Test circuit breaker initial state."""
        cb = CircuitBreakerState()
        assert cb.state == "closed"
        assert cb.failure_count == 0
        assert cb.should_attempt_request() is True

    def test_failure_tracking(self) -> None:
        """Test failure tracking and state transitions."""
        cb = CircuitBreakerState(failure_threshold=2)

        # First failure
        cb.record_failure()
        assert cb.failure_count == 1
        assert cb.state == "closed"
        assert cb.should_attempt_request() is True

        # Second failure - should open circuit
        cb.record_failure()
        assert cb.failure_count == 2
        assert cb.state == "open"
        assert cb.should_attempt_request() is False

    def test_recovery(self) -> None:
        """Test circuit breaker recovery."""
        cb = CircuitBreakerState(failure_threshold=1, recovery_timeout=0.1)

        # Trip circuit breaker
        cb.record_failure()
        assert cb.state == "open"
        assert cb.should_attempt_request() is False

        # Wait for recovery timeout
        time.sleep(0.2)

        # Should transition to half-open
        assert cb.should_attempt_request() is True

        # Success should close circuit
        cb.record_success()
        assert cb.state == "closed"
        assert cb.failure_count == 0


class TestServiceHealthChecker:
    """Test service health checker."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.health_checker = ServiceHealthChecker(timeout=1.0)

    @pytest.mark.asyncio
    async def test_cognito_health_check_success(self) -> None:
        """Test successful Cognito health check."""
        mock_response = {
            "UserPool": {
                "Id": "us-east-1_test123",
                "Name": "test-pool",
            }
        }

        with patch("boto3.client") as mock_client:
            mock_cognito = Mock()
            mock_cognito.describe_user_pool.return_value = mock_response
            mock_client.return_value = mock_cognito

            result = await self.health_checker.check_cognito_health(
                region="us-east-1",
                user_pool_id="us-east-1_test123",
                client_id="test-client",
            )

            assert result.service_name == "cognito"
            assert result.status == ServiceStatus.HEALTHY
            assert result.is_healthy()
            assert "test-pool" in str(result.details)

    @pytest.mark.asyncio
    async def test_cognito_health_check_timeout(self) -> None:
        """Test Cognito health check timeout."""
        with patch("boto3.client") as mock_client:
            mock_cognito = Mock()
            # Simulate timeout by raising asyncio.TimeoutError
            mock_cognito.describe_user_pool.side_effect = TimeoutError(
                "Operation timed out"
            )
            mock_client.return_value = mock_cognito

            result = await self.health_checker.check_cognito_health(
                region="us-east-1",
                user_pool_id="us-east-1_test123",
                client_id="test-client",
                timeout=0.1,
            )

            assert result.service_name == "cognito"
            assert result.status == ServiceStatus.UNHEALTHY
            assert "timed out" in result.message.lower()

    @pytest.mark.asyncio
    async def test_cognito_health_check_no_credentials(self) -> None:
        """Test Cognito health check with no credentials."""
        with patch("boto3.client") as mock_client:
            mock_client.side_effect = NoCredentialsError()

            result = await self.health_checker.check_cognito_health(
                region="us-east-1",
                user_pool_id="us-east-1_test123",
                client_id="test-client",
            )

            assert result.service_name == "cognito"
            assert result.status == ServiceStatus.SKIPPED
            assert "credentials not available" in result.message.lower()

    @pytest.mark.asyncio
    async def test_cognito_health_check_access_denied(self) -> None:
        """Test Cognito health check with access denied."""
        client_error = ClientError(
            error_response={
                "Error": {
                    "Code": "AccessDenied",
                    "Message": "User is not authorized to perform cognito-idp:DescribeUserPool",
                }
            },
            operation_name="DescribeUserPool",
        )

        with patch("boto3.client") as mock_client:
            mock_cognito = Mock()
            mock_cognito.describe_user_pool.side_effect = client_error
            mock_client.return_value = mock_cognito

            result = await self.health_checker.check_cognito_health(
                region="us-east-1",
                user_pool_id="us-east-1_test123",
                client_id="test-client",
            )

            assert result.service_name == "cognito"
            assert result.status == ServiceStatus.DEGRADED
            assert "access denied" in result.message.lower()
            assert result.details["error_code"] == "AccessDenied"

    @pytest.mark.asyncio
    async def test_dynamodb_health_check_success(self) -> None:
        """Test successful DynamoDB health check."""
        mock_response = {
            "Table": {
                "TableName": "test-table",
                "TableStatus": "ACTIVE",
                "ItemCount": 100,
            }
        }

        with patch("boto3.client") as mock_client:
            mock_dynamodb = Mock()
            mock_dynamodb.describe_table.return_value = mock_response
            mock_client.return_value = mock_dynamodb

            result = await self.health_checker.check_dynamodb_health(
                table_name="test-table",
                region="us-east-1",
            )

            assert result.service_name == "dynamodb"
            assert result.status == ServiceStatus.HEALTHY
            assert result.details["table_status"] == "ACTIVE"
            assert result.details["item_count"] == 100

    @pytest.mark.asyncio
    async def test_dynamodb_health_check_creating(self) -> None:
        """Test DynamoDB health check with table creating."""
        mock_response = {
            "Table": {
                "TableName": "test-table",
                "TableStatus": "CREATING",
            }
        }

        with patch("boto3.client") as mock_client:
            mock_dynamodb = Mock()
            mock_dynamodb.describe_table.return_value = mock_response
            mock_client.return_value = mock_dynamodb

            result = await self.health_checker.check_dynamodb_health(
                table_name="test-table",
                region="us-east-1",
            )

            assert result.service_name == "dynamodb"
            assert result.status == ServiceStatus.DEGRADED
            assert "creating" in result.message.lower()

    @pytest.mark.asyncio
    async def test_dynamodb_health_check_not_found(self) -> None:
        """Test DynamoDB health check with table not found."""
        client_error = ClientError(
            error_response={
                "Error": {
                    "Code": "ResourceNotFoundException",
                    "Message": "Requested resource not found",
                }
            },
            operation_name="DescribeTable",
        )

        with patch("boto3.client") as mock_client:
            mock_dynamodb = Mock()
            mock_dynamodb.describe_table.side_effect = client_error
            mock_client.return_value = mock_dynamodb

            result = await self.health_checker.check_dynamodb_health(
                table_name="test-table",
                region="us-east-1",
            )

            assert result.service_name == "dynamodb"
            assert result.status == ServiceStatus.UNHEALTHY
            assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_s3_health_check_success(self) -> None:
        """Test successful S3 health check."""
        with patch("boto3.client") as mock_client:
            mock_s3 = Mock()
            mock_s3.head_bucket.return_value = {}
            mock_client.return_value = mock_s3

            result = await self.health_checker.check_s3_health(
                bucket_name="test-bucket",
                region="us-east-1",
            )

            assert result.service_name == "s3"
            assert result.status == ServiceStatus.HEALTHY
            assert "accessible" in result.message.lower()

    @pytest.mark.asyncio
    async def test_s3_health_check_not_found(self) -> None:
        """Test S3 health check with bucket not found."""
        client_error = ClientError(
            error_response={
                "Error": {
                    "Code": "NoSuchBucket",
                    "Message": "The specified bucket does not exist",
                }
            },
            operation_name="HeadBucket",
        )

        with patch("boto3.client") as mock_client:
            mock_s3 = Mock()
            mock_s3.head_bucket.side_effect = client_error
            mock_client.return_value = mock_s3

            result = await self.health_checker.check_s3_health(
                bucket_name="test-bucket",
                region="us-east-1",
            )

            assert result.service_name == "s3"
            assert result.status == ServiceStatus.UNHEALTHY
            assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self) -> None:
        """Test circuit breaker integration with health checks."""
        client_error = ClientError(
            error_response={
                "Error": {
                    "Code": "ServiceUnavailable",
                    "Message": "Service unavailable",
                }
            },
            operation_name="DescribeUserPool",
        )

        with patch("boto3.client") as mock_client:
            mock_cognito = Mock()
            mock_cognito.describe_user_pool.side_effect = client_error
            mock_client.return_value = mock_cognito

            # Configure circuit breaker with low threshold for testing
            cb = self.health_checker._get_circuit_breaker("cognito")
            cb.failure_threshold = 2

            # First two failures should trip the circuit breaker
            for _ in range(2):
                result = await self.health_checker.check_cognito_health(
                    region="us-east-1",
                    user_pool_id="us-east-1_test123",
                    client_id="test-client",
                )
                assert result.status == ServiceStatus.UNHEALTHY

            # Third attempt should be blocked by circuit breaker
            result = await self.health_checker.check_cognito_health(
                region="us-east-1",
                user_pool_id="us-east-1_test123",
                client_id="test-client",
            )

            assert result.status == ServiceStatus.UNHEALTHY
            assert "circuit breaker is open" in result.message.lower()

    @pytest.mark.asyncio
    async def test_check_all_services(self) -> None:
        """Test checking all services."""
        # Mock configuration
        mock_config = Mock()
        mock_config.get_service_requirements.return_value = {
            "cognito": True,
            "dynamodb": True,
            "s3": True,
            "gemini": False,
        }
        mock_config.cognito.region = "us-east-1"
        mock_config.cognito.user_pool_id = "us-east-1_test123"
        mock_config.cognito.client_id = "test-client"
        mock_config.aws.region = "us-east-1"
        mock_config.dynamodb.table_name = "test-table"
        mock_config.dynamodb.endpoint_url = None
        mock_config.s3.bucket_name = "test-bucket"
        mock_config.s3.ml_models_bucket = "test-models-bucket"
        mock_config.s3.endpoint_url = None
        mock_config.health_check_timeout = 5

        # Mock all the health check methods
        with (
            patch.object(self.health_checker, "check_cognito_health") as mock_cognito,
            patch.object(self.health_checker, "check_dynamodb_health") as mock_dynamodb,
            patch.object(self.health_checker, "check_s3_health") as mock_s3,
        ):

            mock_cognito.return_value = HealthCheckResult(
                service_name="cognito",
                status=ServiceStatus.HEALTHY,
                message="Healthy",
            )
            mock_dynamodb.return_value = HealthCheckResult(
                service_name="dynamodb",
                status=ServiceStatus.HEALTHY,
                message="Healthy",
            )
            mock_s3.return_value = HealthCheckResult(
                service_name="s3",
                status=ServiceStatus.HEALTHY,
                message="Healthy",
            )

            results = await self.health_checker.check_all_services(mock_config)

            assert "cognito" in results
            assert "dynamodb" in results
            assert "s3_uploads" in results
            assert "s3_models" in results

            # Verify all services were checked
            mock_cognito.assert_called_once()
            mock_dynamodb.assert_called_once()
            assert mock_s3.call_count == 2  # Called for both buckets

    def test_get_overall_health(self) -> None:
        """Test overall health determination."""
        # All healthy
        results = {
            "cognito": HealthCheckResult("cognito", ServiceStatus.HEALTHY, "OK"),
            "dynamodb": HealthCheckResult("dynamodb", ServiceStatus.HEALTHY, "OK"),
        }
        assert self.health_checker.get_overall_health(results) == ServiceStatus.HEALTHY

        # One degraded
        results["s3"] = HealthCheckResult("s3", ServiceStatus.DEGRADED, "Slow")
        assert self.health_checker.get_overall_health(results) == ServiceStatus.DEGRADED

        # One unhealthy
        results["cognito"] = HealthCheckResult(
            "cognito", ServiceStatus.UNHEALTHY, "Failed"
        )
        assert (
            self.health_checker.get_overall_health(results) == ServiceStatus.UNHEALTHY
        )

        # All skipped
        results = {
            "cognito": HealthCheckResult("cognito", ServiceStatus.SKIPPED, "Skipped"),
            "dynamodb": HealthCheckResult("dynamodb", ServiceStatus.SKIPPED, "Skipped"),
        }
        assert self.health_checker.get_overall_health(results) == ServiceStatus.HEALTHY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
