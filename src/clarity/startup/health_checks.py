"""CLARITY Service Health Checks.

Independent health checks for each external service with circuit breakers
and detailed error reporting.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
import logging
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class ServiceStatus(StrEnum):
    """Service health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    SKIPPED = "skipped"


@dataclass
class HealthCheckResult:
    """Result of a service health check."""

    service_name: str
    status: ServiceStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    response_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    error: Exception | None = None

    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self.status == ServiceStatus.HEALTHY

    def is_usable(self) -> bool:
        """Check if service is usable (healthy or degraded)."""
        return self.status in {ServiceStatus.HEALTHY, ServiceStatus.DEGRADED}


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for a service."""

    failure_count: int = 0
    last_failure_time: float = 0.0
    state: str = "closed"  # closed, open, half-open
    failure_threshold: int = 3
    recovery_timeout: float = 60.0  # seconds

    def should_attempt_request(self) -> bool:
        """Check if we should attempt a request."""
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        # half-open
        return True

    def record_success(self) -> None:
        """Record successful request."""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self) -> None:
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class ServiceHealthChecker:
    """Service health checker with circuit breakers."""

    def __init__(self, timeout: float = 5.0) -> None:
        """Initialize health checker.

        Args:
            timeout: Default timeout for health checks in seconds
        """
        self.timeout = timeout
        self.circuit_breakers: dict[str, CircuitBreakerState] = {}

    def _get_circuit_breaker(self, service_name: str) -> CircuitBreakerState:
        """Get or create circuit breaker for service."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreakerState()
        return self.circuit_breakers[service_name]

    async def check_cognito_health(
        self,
        region: str,
        user_pool_id: str,
        client_id: str,
        timeout: float | None = None,
    ) -> HealthCheckResult:
        """Check AWS Cognito health."""
        service_name = "cognito"
        start_time = time.time()
        circuit_breaker = self._get_circuit_breaker(service_name)

        try:
            if not circuit_breaker.should_attempt_request():
                return HealthCheckResult(
                    service_name=service_name,
                    status=ServiceStatus.UNHEALTHY,
                    message="Circuit breaker is open - too many recent failures",
                    details={"circuit_breaker_state": circuit_breaker.state},
                    response_time_ms=0.0,
                )

            # Create Cognito client
            client = boto3.client("cognito-idp", region_name=region)

            # Test connectivity by describing user pool
            actual_timeout = timeout or self.timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(client.describe_user_pool, UserPoolId=user_pool_id),
                timeout=actual_timeout,
            )

            response_time = (time.time() - start_time) * 1000

            # If we reach here, the response was successful
            circuit_breaker.record_success()
            return HealthCheckResult(
                service_name=service_name,
                status=ServiceStatus.HEALTHY,
                message="Cognito service is healthy",
                details={
                    "user_pool_id": user_pool_id,
                    "user_pool_name": response["UserPool"].get("Name", "unknown"),
                    "region": region,
                },
                response_time_ms=response_time,
            )

        except TimeoutError:
            circuit_breaker.record_failure()
            return HealthCheckResult(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                message=f"Cognito health check timed out after {timeout or self.timeout}s",
                response_time_ms=(time.time() - start_time) * 1000,
            )

        except NoCredentialsError as e:
            return HealthCheckResult(
                service_name=service_name,
                status=ServiceStatus.SKIPPED,
                message="AWS credentials not available",
                details={"error": str(e)},
                response_time_ms=(time.time() - start_time) * 1000,
                error=e,
            )

        except ClientError as e:
            circuit_breaker = self._get_circuit_breaker(service_name)
            circuit_breaker.record_failure()

            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            # Some errors indicate service availability issues
            if error_code in {"ResourceNotFoundException", "UserPoolNotFound"}:
                status = ServiceStatus.UNHEALTHY
                message = f"Cognito User Pool not found: {user_pool_id}"
            elif error_code in {"UnauthorizedOperation", "AccessDenied"}:
                status = ServiceStatus.DEGRADED
                message = f"Cognito access denied - check permissions: {error_message}"
            else:
                status = ServiceStatus.UNHEALTHY
                message = f"Cognito error ({error_code}): {error_message}"

            return HealthCheckResult(
                service_name=service_name,
                status=status,
                message=message,
                details={
                    "error_code": error_code,
                    "error_message": error_message,
                    "user_pool_id": user_pool_id,
                },
                response_time_ms=(time.time() - start_time) * 1000,
                error=e,
            )

        except Exception as e:  # noqa: BLE001 - Health check needs to catch all
            circuit_breaker = self._get_circuit_breaker(service_name)
            circuit_breaker.record_failure()
            return HealthCheckResult(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                message=f"Cognito health check failed: {e!s}",
                response_time_ms=(time.time() - start_time) * 1000,
                error=e,
            )

    async def check_dynamodb_health(
        self,
        table_name: str,
        region: str,
        endpoint_url: str | None = None,
        timeout: float | None = None,
    ) -> HealthCheckResult:
        """Check DynamoDB health."""
        service_name = "dynamodb"
        start_time = time.time()
        circuit_breaker = self._get_circuit_breaker(service_name)

        try:
            if not circuit_breaker.should_attempt_request():
                return HealthCheckResult(
                    service_name=service_name,
                    status=ServiceStatus.UNHEALTHY,
                    message="Circuit breaker is open - too many recent failures",
                    details={"circuit_breaker_state": circuit_breaker.state},
                    response_time_ms=0.0,
                )

            # Create DynamoDB client
            client_kwargs = {"region_name": region}
            if endpoint_url:
                client_kwargs["endpoint_url"] = endpoint_url

            client = boto3.client("dynamodb", **client_kwargs)  # type: ignore[call-overload]

            # Test connectivity by describing table
            actual_timeout = timeout or self.timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(client.describe_table, TableName=table_name),
                timeout=actual_timeout,
            )

            response_time = (time.time() - start_time) * 1000

            # Check table status
            table_status = response.get("Table", {}).get("TableStatus", "UNKNOWN")

            if table_status == "ACTIVE":
                circuit_breaker.record_success()
                return HealthCheckResult(
                    service_name=service_name,
                    status=ServiceStatus.HEALTHY,
                    message="DynamoDB table is active and healthy",
                    details={
                        "table_name": table_name,
                        "table_status": table_status,
                        "region": region,
                        "item_count": response.get("Table", {}).get("ItemCount", 0),
                    },
                    response_time_ms=response_time,
                )
            if table_status in {"CREATING", "UPDATING"}:
                return HealthCheckResult(
                    service_name=service_name,
                    status=ServiceStatus.DEGRADED,
                    message=f"DynamoDB table is {table_status.lower()} - service may be slow",
                    details={
                        "table_name": table_name,
                        "table_status": table_status,
                        "region": region,
                    },
                    response_time_ms=response_time,
                )
            circuit_breaker.record_failure()
            return HealthCheckResult(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                message=f"DynamoDB table is in {table_status} state",
                details={
                    "table_name": table_name,
                    "table_status": table_status,
                    "region": region,
                },
                response_time_ms=response_time,
            )

        except TimeoutError:
            circuit_breaker.record_failure()
            return HealthCheckResult(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                message=f"DynamoDB health check timed out after {timeout or self.timeout}s",
                response_time_ms=(time.time() - start_time) * 1000,
            )

        except NoCredentialsError as e:
            return HealthCheckResult(
                service_name=service_name,
                status=ServiceStatus.SKIPPED,
                message="AWS credentials not available",
                details={"error": str(e)},
                response_time_ms=(time.time() - start_time) * 1000,
                error=e,
            )

        except ClientError as e:
            circuit_breaker = self._get_circuit_breaker(service_name)
            circuit_breaker.record_failure()

            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            if error_code == "ResourceNotFoundException":
                status = ServiceStatus.UNHEALTHY
                message = f"DynamoDB table not found: {table_name}"
            elif error_code in {"UnauthorizedOperation", "AccessDenied"}:
                status = ServiceStatus.DEGRADED
                message = f"DynamoDB access denied - check permissions: {error_message}"
            else:
                status = ServiceStatus.UNHEALTHY
                message = f"DynamoDB error ({error_code}): {error_message}"

            return HealthCheckResult(
                service_name=service_name,
                status=status,
                message=message,
                details={
                    "error_code": error_code,
                    "error_message": error_message,
                    "table_name": table_name,
                },
                response_time_ms=(time.time() - start_time) * 1000,
                error=e,
            )

        except Exception as e:  # noqa: BLE001 - Health check needs to catch all
            circuit_breaker = self._get_circuit_breaker(service_name)
            circuit_breaker.record_failure()
            return HealthCheckResult(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                message=f"DynamoDB health check failed: {e!s}",
                response_time_ms=(time.time() - start_time) * 1000,
                error=e,
            )

    async def check_s3_health(
        self,
        bucket_name: str,
        region: str,
        endpoint_url: str | None = None,
        timeout: float | None = None,
    ) -> HealthCheckResult:
        """Check S3 bucket health."""
        service_name = "s3"
        start_time = time.time()
        circuit_breaker = self._get_circuit_breaker(service_name)

        try:
            if not circuit_breaker.should_attempt_request():
                return HealthCheckResult(
                    service_name=service_name,
                    status=ServiceStatus.UNHEALTHY,
                    message="Circuit breaker is open - too many recent failures",
                    details={"circuit_breaker_state": circuit_breaker.state},
                    response_time_ms=0.0,
                )

            # Create S3 client
            client_kwargs = {"region_name": region}
            if endpoint_url:
                client_kwargs["endpoint_url"] = endpoint_url

            client = boto3.client("s3", **client_kwargs)  # type: ignore[call-overload]

            # Test connectivity by getting bucket location
            actual_timeout = timeout or self.timeout
            await asyncio.wait_for(
                asyncio.to_thread(client.head_bucket, Bucket=bucket_name),
                timeout=actual_timeout,
            )

            response_time = (time.time() - start_time) * 1000

            circuit_breaker.record_success()
            return HealthCheckResult(
                service_name=service_name,
                status=ServiceStatus.HEALTHY,
                message="S3 bucket is accessible and healthy",
                details={
                    "bucket_name": bucket_name,
                    "region": region,
                },
                response_time_ms=response_time,
            )

        except TimeoutError:
            circuit_breaker.record_failure()
            return HealthCheckResult(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                message=f"S3 health check timed out after {timeout or self.timeout}s",
                response_time_ms=(time.time() - start_time) * 1000,
            )

        except NoCredentialsError as e:
            return HealthCheckResult(
                service_name=service_name,
                status=ServiceStatus.SKIPPED,
                message="AWS credentials not available",
                details={"error": str(e)},
                response_time_ms=(time.time() - start_time) * 1000,
                error=e,
            )

        except ClientError as e:
            circuit_breaker = self._get_circuit_breaker(service_name)
            circuit_breaker.record_failure()

            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            if error_code == "NoSuchBucket":
                status = ServiceStatus.UNHEALTHY
                message = f"S3 bucket not found: {bucket_name}"
            elif error_code in {"AccessDenied", "Forbidden"}:
                status = ServiceStatus.DEGRADED
                message = f"S3 access denied - check permissions: {error_message}"
            else:
                status = ServiceStatus.UNHEALTHY
                message = f"S3 error ({error_code}): {error_message}"

            return HealthCheckResult(
                service_name=service_name,
                status=status,
                message=message,
                details={
                    "error_code": error_code,
                    "error_message": error_message,
                    "bucket_name": bucket_name,
                },
                response_time_ms=(time.time() - start_time) * 1000,
                error=e,
            )

        except Exception as e:  # noqa: BLE001 - Health check needs to catch all
            circuit_breaker = self._get_circuit_breaker(service_name)
            circuit_breaker.record_failure()
            return HealthCheckResult(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                message=f"S3 health check failed: {e!s}",
                response_time_ms=(time.time() - start_time) * 1000,
                error=e,
            )

    async def check_all_services(
        self,
        config: Any,
        skip_services: set[str] | None = None,
    ) -> dict[str, HealthCheckResult]:
        """Check health of all configured services.

        Args:
            config: Application configuration
            skip_services: Set of service names to skip

        Returns:
            Dictionary of service names to health check results
        """
        skip_services = skip_services or set()
        results = {}

        # Determine which services to check
        service_requirements = config.get_service_requirements()

        # Check Cognito
        if "cognito" not in skip_services and service_requirements.get(
            "cognito", False
        ):
            results["cognito"] = await self.check_cognito_health(
                region=config.cognito.region or config.aws.region,
                user_pool_id=config.cognito.user_pool_id,
                client_id=config.cognito.client_id,
                timeout=config.health_check_timeout,
            )
        elif "cognito" not in skip_services:
            results["cognito"] = HealthCheckResult(
                service_name="cognito",
                status=ServiceStatus.SKIPPED,
                message="Cognito not required or auth disabled",
            )

        # Check DynamoDB
        if "dynamodb" not in skip_services and service_requirements.get(
            "dynamodb", False
        ):
            results["dynamodb"] = await self.check_dynamodb_health(
                table_name=config.dynamodb.table_name,
                region=config.aws.region,
                endpoint_url=config.dynamodb.endpoint_url or None,
                timeout=config.health_check_timeout,
            )
        elif "dynamodb" not in skip_services:
            results["dynamodb"] = HealthCheckResult(
                service_name="dynamodb",
                status=ServiceStatus.SKIPPED,
                message="DynamoDB not required or using mock",
            )

        # Check S3
        if "s3" not in skip_services and service_requirements.get("s3", False):
            # Check both buckets
            results["s3_uploads"] = await self.check_s3_health(
                bucket_name=config.s3.bucket_name,
                region=config.aws.region,
                endpoint_url=config.s3.endpoint_url or None,
                timeout=config.health_check_timeout,
            )

            results["s3_models"] = await self.check_s3_health(
                bucket_name=config.s3.ml_models_bucket,
                region=config.aws.region,
                endpoint_url=config.s3.endpoint_url or None,
                timeout=config.health_check_timeout,
            )
        elif "s3" not in skip_services:
            results["s3"] = HealthCheckResult(
                service_name="s3",
                status=ServiceStatus.SKIPPED,
                message="S3 not required or using mock",
            )

        return results

    def get_overall_health(
        self, results: dict[str, HealthCheckResult]
    ) -> ServiceStatus:
        """Determine overall system health from individual service results."""
        if not results:
            return ServiceStatus.UNKNOWN

        statuses = [result.status for result in results.values()]

        # If any service is unhealthy, system is unhealthy
        if ServiceStatus.UNHEALTHY in statuses:
            return ServiceStatus.UNHEALTHY

        # If any service is degraded, system is degraded
        if ServiceStatus.DEGRADED in statuses:
            return ServiceStatus.DEGRADED

        # If all services are healthy or skipped, system is healthy
        return ServiceStatus.HEALTHY
