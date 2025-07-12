"""Enhanced Feature Flag System with Auto-Refresh Capabilities.

This module extends the feature flag system with background refresh,
pub/sub support, circuit breaker pattern, and comprehensive metrics.
"""

import asyncio
from collections.abc import Generator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from enum import StrEnum
import logging
import threading
import time
from typing import TypeVar

import aioredis  # type: ignore[import-not-found]
from circuitbreaker import CircuitBreaker, CircuitBreakerError
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field

from clarity.core.config_provider import ConfigProvider
from clarity.core.feature_flags import FeatureFlagConfig
from clarity.core.feature_flags import (
    FeatureFlagManager as BaseFeatureFlagManager,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Prometheus metrics for feature flag operations
FEATURE_FLAG_REFRESH_SUCCESS = Counter(
    "feature_flag_refresh_success_total",
    "Total number of successful feature flag refreshes",
)
FEATURE_FLAG_REFRESH_FAILURE = Counter(
    "feature_flag_refresh_failure_total",
    "Total number of failed feature flag refreshes",
    ["error_type"],
)
FEATURE_FLAG_STALE_CONFIG = Gauge(
    "feature_flag_stale_config",
    "Whether the feature flag configuration is stale (1) or fresh (0)",
)
FEATURE_FLAG_LAST_REFRESH = Gauge(
    "feature_flag_last_refresh_timestamp",
    "Timestamp of the last successful feature flag refresh",
)
FEATURE_FLAG_REFRESH_DURATION = Histogram(
    "feature_flag_refresh_duration_seconds",
    "Time taken to refresh feature flags",
)
FEATURE_FLAG_CIRCUIT_BREAKER_STATE = Gauge(
    "feature_flag_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half-open)",
)


class RefreshMode(StrEnum):
    """Feature flag refresh modes."""

    PERIODIC = "periodic"
    PUBSUB = "pubsub"
    BOTH = "both"
    NONE = "none"


class EnhancedFeatureFlagConfig(BaseModel):
    """Enhanced configuration for feature flag system."""

    refresh_interval_seconds: int = Field(default=60, ge=10)
    refresh_mode: RefreshMode = Field(default=RefreshMode.BOTH)
    redis_url: str = Field(default="redis://localhost:6379")
    pubsub_channel: str = Field(default="feature_flags:config-changed")
    circuit_breaker_failure_threshold: int = Field(default=3, ge=1)
    circuit_breaker_recovery_timeout: int = Field(default=30, ge=10)
    stale_config_threshold_seconds: int = Field(default=300, ge=60)
    enable_metrics: bool = Field(default=True)


class EnhancedFeatureFlagManager(BaseFeatureFlagManager):
    """Feature flag manager with auto-refresh, pub/sub, and circuit breaker capabilities."""

    def __init__(
        self,
        config_provider: ConfigProvider | None = None,
        enhanced_config: EnhancedFeatureFlagConfig | None = None,
    ) -> None:
        """Initialize enhanced feature flag manager.

        Args:
            config_provider: Configuration provider for remote config store
            enhanced_config: Enhanced configuration settings
        """
        super().__init__()

        self._config_provider = config_provider
        self._enhanced_config = enhanced_config or EnhancedFeatureFlagConfig()

        # Thread safety
        self._lock = threading.RLock()
        self._refresh_task: asyncio.Task[None] | None = None
        self._pubsub_task: asyncio.Task[None] | None = None
        self._shutdown_event = threading.Event()

        # Tracking
        self._last_refresh_time: datetime | None = None
        self._refresh_failures = 0

        # Circuit breaker for config store access
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=self._enhanced_config.circuit_breaker_failure_threshold,
            recovery_timeout=self._enhanced_config.circuit_breaker_recovery_timeout,
            expected_exception=Exception,
        )

        # Redis client for pub/sub
        self._redis_client: aioredis.Redis | None = None

        # Start background tasks if needed
        if self._enhanced_config.refresh_mode != RefreshMode.NONE:
            self._start_background_tasks()

    def _start_background_tasks(self) -> None:
        """Start background refresh and pub/sub tasks."""
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, create one in a thread

            def run_event_loop() -> None:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                new_loop.run_until_complete(self._async_background_tasks())

            thread = threading.Thread(target=run_event_loop, daemon=True)
            thread.start()
        else:
            # Event loop is running, schedule tasks
            if self._enhanced_config.refresh_mode in {
                RefreshMode.PERIODIC,
                RefreshMode.BOTH,
            }:
                self._refresh_task = loop.create_task(self._periodic_refresh_loop())

            if self._enhanced_config.refresh_mode in {
                RefreshMode.PUBSUB,
                RefreshMode.BOTH,
            }:
                self._pubsub_task = loop.create_task(self._pubsub_listener())

    async def _async_background_tasks(self) -> None:
        """Run background tasks in async context."""
        tasks = []

        if self._enhanced_config.refresh_mode in {
            RefreshMode.PERIODIC,
            RefreshMode.BOTH,
        }:
            tasks.append(self._periodic_refresh_loop())

        if self._enhanced_config.refresh_mode in {RefreshMode.PUBSUB, RefreshMode.BOTH}:
            tasks.append(self._pubsub_listener())

        if tasks:
            await asyncio.gather(*tasks)

    async def _periodic_refresh_loop(self) -> None:
        """Background task for periodic feature flag refresh."""
        logger.info(
            "Starting periodic feature flag refresh, interval=%ds",
            self._enhanced_config.refresh_interval_seconds,
        )

        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self._enhanced_config.refresh_interval_seconds)
                await self.refresh_async()
            except asyncio.CancelledError:
                logger.info("Periodic refresh task cancelled")
                break
            except Exception as e:
                logger.exception("Error in periodic refresh loop: %s", e)
                await asyncio.sleep(10)  # Back off on errors

    async def _pubsub_listener(self) -> None:
        """Background task for pub/sub config change listener."""
        logger.info(
            "Starting pub/sub listener on channel: %s",
            self._enhanced_config.pubsub_channel,
        )

        while not self._shutdown_event.is_set():
            try:
                if not self._redis_client:
                    self._redis_client = await aioredis.create_redis_pool(
                        self._enhanced_config.redis_url
                    )

                # Subscribe to config change channel
                channel = (
                    await self._redis_client.subscribe(
                        self._enhanced_config.pubsub_channel
                    )
                )[0]

                # Listen for messages
                while await channel.wait_message():
                    msg = await channel.get()
                    if msg:
                        logger.info("Received config change notification: %s", msg)
                        await self.refresh_async()

            except asyncio.CancelledError:
                logger.info("Pub/sub listener task cancelled")
                break
            except Exception as e:
                logger.exception("Error in pub/sub listener: %s", e)
                await asyncio.sleep(10)  # Back off on errors

    def refresh(self) -> bool:
        """Synchronously refresh feature flags from config store.

        Returns:
            True if refresh successful, False otherwise
        """
        try:
            # Create event loop if needed
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(self.refresh_async())
                loop.close()
                return result
            else:
                # Run in existing loop
                future = asyncio.ensure_future(self.refresh_async())
                return loop.run_until_complete(future)
        except Exception as e:
            logger.exception("Error in synchronous refresh: %s", e)
            return False

    async def refresh_async(self) -> bool:
        """Asynchronously refresh feature flags from config store.

        Returns:
            True if refresh successful, False otherwise
        """
        start_time = time.time()

        try:
            with self._refresh_metrics():
                # Check circuit breaker
                if self._circuit_breaker.current_state == "open":
                    logger.warning("Circuit breaker is open, skipping refresh")
                    self._update_circuit_breaker_metric()
                    msg = "Config store circuit breaker is open"
                    raise CircuitBreakerError(msg)

                # Attempt to fetch new configuration
                new_config = await self._fetch_config_from_store()

                if new_config:
                    with self._lock:
                        self.config = new_config
                        self.clear_cache()
                        self._last_refresh_time = datetime.now(UTC)
                        self._refresh_failures = 0

                    # Update metrics
                    FEATURE_FLAG_REFRESH_SUCCESS.inc()
                    FEATURE_FLAG_LAST_REFRESH.set(time.time())
                    FEATURE_FLAG_STALE_CONFIG.set(0)
                    self._update_circuit_breaker_metric()

                    logger.info(
                        "Successfully refreshed feature flags, flags=%d",
                        len(new_config.flags),
                    )
                    return True
                msg = "Empty configuration received from store"
                raise ValueError(msg)

        except CircuitBreakerError:
            self._handle_refresh_failure("circuit_breaker_open")
            return False
        except Exception as e:
            logger.exception("Failed to refresh feature flags: %s", e)
            self._handle_refresh_failure(type(e).__name__)

            # Update circuit breaker
            with suppress(Exception):
                self._circuit_breaker(lambda: None)()

            return False
        finally:
            # Record refresh duration
            duration = time.time() - start_time
            FEATURE_FLAG_REFRESH_DURATION.observe(duration)

    async def _fetch_config_from_store(self) -> FeatureFlagConfig | None:
        """Fetch configuration from remote config store.

        Returns:
            Feature flag configuration or None if unavailable
        """
        if not self._config_provider:
            logger.debug("No config provider available, using defaults")
            return None

        try:
            # Simulate fetching from config store (AWS Parameter Store, DynamoDB, etc.)
            # In real implementation, this would call the actual config store API

            # For now, return mock configuration
            # TODO: Implement actual config store integration
            return self._load_configuration()

        except Exception as e:
            logger.exception("Error fetching config from store: %s", e)
            raise

    def _handle_refresh_failure(self, error_type: str) -> None:
        """Handle refresh failure and update metrics."""
        with self._lock:
            self._refresh_failures += 1

            # Check if config is stale
            if self._last_refresh_time:
                age = (datetime.now(UTC) - self._last_refresh_time).total_seconds()
                if age > self._enhanced_config.stale_config_threshold_seconds:
                    FEATURE_FLAG_STALE_CONFIG.set(1)
                    logger.warning("Feature flag configuration is stale, age=%ds", age)

        FEATURE_FLAG_REFRESH_FAILURE.labels(error_type=error_type).inc()
        self._update_circuit_breaker_metric()

    def _update_circuit_breaker_metric(self) -> None:
        """Update circuit breaker state metric."""
        state_map = {"closed": 0, "open": 1, "half-open": 2}
        state_value = state_map.get(self._circuit_breaker.current_state, -1)
        FEATURE_FLAG_CIRCUIT_BREAKER_STATE.set(state_value)

    @contextmanager
    def _refresh_metrics(self) -> Generator[None, None, None]:
        """Context manager for refresh metrics."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            if self._enhanced_config.enable_metrics:
                FEATURE_FLAG_REFRESH_DURATION.observe(duration)

    def get_config_age_seconds(self) -> float | None:
        """Get age of current configuration in seconds.

        Returns:
            Age in seconds or None if never refreshed
        """
        with self._lock:
            if self._last_refresh_time:
                return (datetime.now(UTC) - self._last_refresh_time).total_seconds()
            return None

    def is_config_stale(self) -> bool:
        """Check if current configuration is stale.

        Returns:
            True if config is stale, False otherwise
        """
        age = self.get_config_age_seconds()
        if age is None:
            return True  # Never refreshed
        return age > self._enhanced_config.stale_config_threshold_seconds

    def get_circuit_breaker_state(self) -> str:
        """Get current circuit breaker state.

        Returns:
            Circuit breaker state: 'closed', 'open', or 'half-open'
        """
        return str(self._circuit_breaker.current_state)

    def shutdown(self) -> None:
        """Shutdown background tasks and cleanup resources."""
        logger.info("Shutting down enhanced feature flag manager")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel async tasks
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()

        if self._pubsub_task and not self._pubsub_task.done():
            self._pubsub_task.cancel()

        # Close Redis connection
        if self._redis_client:
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self._redis_client.close())
            except Exception as e:
                logger.exception("Error closing Redis connection: %s", e)

    def __enter__(self) -> "EnhancedFeatureFlagManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object) -> None:
        """Context manager exit."""
        self.shutdown()


# Global enhanced instance
_enhanced_feature_flag_manager: EnhancedFeatureFlagManager | None = None


def get_enhanced_feature_flag_manager(
    config_provider: ConfigProvider | None = None,
    enhanced_config: EnhancedFeatureFlagConfig | None = None,
) -> EnhancedFeatureFlagManager:
    """Get or create the global enhanced feature flag manager.

    Args:
        config_provider: Configuration provider for remote config store
        enhanced_config: Enhanced configuration settings

    Returns:
        Enhanced feature flag manager instance
    """
    global _enhanced_feature_flag_manager
    if _enhanced_feature_flag_manager is None:
        _enhanced_feature_flag_manager = EnhancedFeatureFlagManager(
            config_provider=config_provider,
            enhanced_config=enhanced_config,
        )
    return _enhanced_feature_flag_manager
