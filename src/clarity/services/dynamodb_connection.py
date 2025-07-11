"""DynamoDB Connection Management - Following SOLID principles.

Single Responsibility: Only manages DynamoDB connections
Open/Closed: Extensible through configuration, closed for modification
Liskov Substitution: Can be replaced with mock for testing
Interface Segregation: Small, focused interface
Dependency Inversion: Depends on boto3 abstraction, not concrete implementation
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from threading import Lock, Semaphore
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError
from mypy_boto3_dynamodb.service_resource import DynamoDBServiceResource

logger = logging.getLogger(__name__)


class DynamoDBConnectionError(Exception):
    """Base exception for connection errors."""


class RetryableConnectionError(DynamoDBConnectionError):
    """Error that can be retried."""


class ConnectionPoolExhaustedError(DynamoDBConnectionError):
    """Connection pool has no available connections."""


@dataclass
class ConnectionConfig:
    """Configuration for DynamoDB connections.

    Follows Builder pattern for clean configuration.
    """

    region: str
    endpoint_url: str | None = None
    max_pool_size: int = 50
    connection_timeout: float = 30.0
    pool_timeout: float = 5.0
    health_check_interval: int = 60
    enable_metrics: bool = True
    enable_auto_failover: bool = False
    failover_regions: list[str] = field(default_factory=list)
    retry_config: dict[str, Any] = field(
        default_factory=lambda: {
            "max_attempts": 3,
            "base_delay": 0.1,
            "max_delay": 20.0,
            "exponential_base": 2,
        }
    )
    circuit_breaker_config: dict[str, Any] = field(
        default_factory=lambda: {
            "failure_threshold": 5,
            "recovery_timeout": 30,
            "expected_exception": ClientError,
        }
    )

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.region:
            msg = "region is required"
            raise ValueError(msg)
        if self.max_pool_size <= 0:
            msg = "max_pool_size must be positive"
            raise ValueError(msg)
        if self.connection_timeout <= 0:
            msg = "connection_timeout must be positive"
            raise ValueError(msg)


@dataclass
class HealthStatus:
    """Health check result."""

    is_healthy: bool
    latency_ms: float
    last_check_time: float
    error_message: str | None = None


@dataclass
class ConnectionMetrics:
    """Connection pool metrics."""

    total_connections: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    active_connections: int = 0
    average_connection_time_ms: float = 0.0
    circuit_breaker_trips: int = 0


class CircuitBreaker:
    """Simple circuit breaker implementation."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(self, failure_threshold: int, recovery_timeout: float) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = self.CLOSED
        self._lock = Lock()

    def call(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == self.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = self.HALF_OPEN
                else:
                    msg = f"Circuit breaker is {self.state}"
                    raise ConnectionPoolExhaustedError(msg)

        try:
            result = func(*args, **kwargs)
            with self._lock:
                if self.state == self.HALF_OPEN:
                    self.state = self.CLOSED
                self.failure_count = 0
            return result
        except Exception:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = self.OPEN
            raise


class DynamoDBConnection:
    """Manages DynamoDB connections with pooling and resilience.

    Implements:
    - Connection pooling
    - Circuit breaker pattern
    - Exponential backoff retry
    - Health checking
    - Metrics collection
    - Regional failover
    """

    def __init__(self, config: ConnectionConfig) -> None:
        self.config = config
        self._resource: DynamoDBServiceResource | None = None
        self._connection_lock = Lock()
        self._pool = Semaphore(config.max_pool_size)
        self._active_connections = 0
        self._metrics = ConnectionMetrics()
        self._circuit_breaker = CircuitBreaker(
            config.circuit_breaker_config["failure_threshold"],
            config.circuit_breaker_config["recovery_timeout"],
        )
        self._current_region = config.region
        self._last_health_check = 0.0
        self._is_connected = False

        logger.info(
            "DynamoDB connection manager initialized for region: %s", config.region
        )

    @property
    def is_connected(self) -> bool:
        """Check if connection is established."""
        return self._is_connected

    @property
    def current_region(self) -> str:
        """Get current active region."""
        return self._current_region

    @property
    def active_connections(self) -> int:
        """Get number of active connections."""
        return self._active_connections

    def get_resource(self) -> DynamoDBServiceResource:
        """Get DynamoDB resource with lazy initialization."""
        if self._resource is None:
            self._resource = self._create_connection()
        return self._resource

    def _create_connection(self) -> DynamoDBServiceResource:
        """Create new DynamoDB connection with retry logic."""
        start_time = time.time()
        last_error = None

        # Try primary region first
        regions_to_try = [self.config.region]
        if self.config.enable_auto_failover:
            regions_to_try.extend(self.config.failover_regions)

        for region in regions_to_try:
            for attempt in range(self.config.retry_config["max_attempts"]):
                try:
                    # Use circuit breaker
                    resource: DynamoDBServiceResource = self._circuit_breaker.call(
                        self._connect_to_region, region
                    )

                    # Update metrics
                    self._metrics.total_connections += 1
                    self._metrics.successful_connections += 1
                    connection_time = (time.time() - start_time) * 1000
                    self._update_average_connection_time(connection_time)

                    self._current_region = region
                    self._is_connected = True

                    logger.info("Connected to DynamoDB in region: %s", region)
                    return resource

                except ConnectionPoolExhaustedError:
                    # Circuit breaker is open, re-raise as-is
                    raise
                except ClientError as e:
                    last_error = e
                    self._metrics.failed_connections += 1

                    if e.response["Error"]["Code"] in {
                        "ServiceUnavailable",
                        "ThrottlingException",
                    }:
                        # Exponential backoff
                        delay = min(
                            self.config.retry_config["base_delay"]
                            * (self.config.retry_config["exponential_base"] ** attempt),
                            self.config.retry_config["max_delay"],
                        )
                        time.sleep(delay)
                        continue
                    msg = f"Failed to connect: {e}"
                    raise RetryableConnectionError(msg) from e

        msg = f"Failed to connect after all attempts: {last_error}"
        raise RetryableConnectionError(msg)

    def _connect_to_region(self, region: str) -> DynamoDBServiceResource:
        """Connect to specific region."""
        return boto3.resource(
            "dynamodb", region_name=region, endpoint_url=self.config.endpoint_url
        )

    def acquire_connection(self) -> DynamoDBServiceResource:
        """Acquire connection from pool."""
        acquired = self._pool.acquire(timeout=self.config.pool_timeout)
        if not acquired:
            msg = "Connection pool timeout"
            raise ConnectionPoolExhaustedError(msg)

        with self._connection_lock:
            self._active_connections += 1
            self._metrics.active_connections = self._active_connections

        return self.get_resource()

    def release_connection(self, connection: Any) -> None:
        """Release connection back to pool."""
        self._pool.release()
        with self._connection_lock:
            self._active_connections -= 1
            self._metrics.active_connections = self._active_connections

    def check_health(self) -> HealthStatus:
        """Check connection health."""
        start_time = time.time()

        try:
            # Simple health check - list tables
            resource = self.get_resource()
            list(resource.tables.limit(1))

            latency_ms = (time.time() - start_time) * 1000
            self._last_health_check = time.time()

            return HealthStatus(
                is_healthy=True,
                latency_ms=latency_ms,
                last_check_time=self._last_health_check,
            )
        except (
            Exception
        ) as e:
            return HealthStatus(
                is_healthy=False,
                latency_ms=0.0,
                last_check_time=time.time(),
                error_message=str(e),
            )

    def get_metrics(self) -> ConnectionMetrics:
        """Get connection metrics."""
        return ConnectionMetrics(
            total_connections=self._metrics.total_connections,
            successful_connections=self._metrics.successful_connections,
            failed_connections=self._metrics.failed_connections,
            active_connections=self._metrics.active_connections,
            average_connection_time_ms=self._metrics.average_connection_time_ms,
            circuit_breaker_trips=self._metrics.circuit_breaker_trips,
        )

    def _update_average_connection_time(self, new_time_ms: float) -> None:
        """Update running average of connection time."""
        if self._metrics.successful_connections == 1:
            self._metrics.average_connection_time_ms = new_time_ms
        else:
            # Running average
            n = self._metrics.successful_connections
            old_avg = self._metrics.average_connection_time_ms
            self._metrics.average_connection_time_ms = (
                (n - 1) * old_avg + new_time_ms
            ) / n

    def shutdown(self, grace_period_seconds: float = 5.0) -> None:
        """Gracefully shutdown connections."""
        logger.info("Shutting down DynamoDB connection manager...")

        # Wait for active connections to complete
        start_time = time.time()
        while (
            self._active_connections > 0
            and (time.time() - start_time) < grace_period_seconds
        ):
            time.sleep(0.1)

        # Force close if needed
        if self._active_connections > 0:
            logger.warning(
                "Forcing shutdown with %d active connections", self._active_connections
            )

        self._is_connected = False
        self._resource = None
        self._active_connections = 0

        logger.info("DynamoDB connection manager shutdown complete")
