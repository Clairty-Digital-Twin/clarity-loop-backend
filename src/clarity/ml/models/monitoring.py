"""ML Model Performance Monitoring Service.

Provides comprehensive monitoring, metrics collection, and alerting for ML models.
Supports Prometheus metrics, custom dashboards, and real-time performance tracking.
"""

import asyncio
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
import time
from typing import Any, TypeVar

from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server
from pydantic import BaseModel

from clarity.ml.models.manager import ModelManager

logger = logging.getLogger(__name__)

# Constants
MODEL_ID_PARTS = 2  # Expected parts in model:version format
MIN_ERROR_SAMPLE_SIZE = 10  # Minimum errors for rate calculation
HTTP_OK = 200  # HTTP success status code


@dataclass
class ModelInferenceMetric:
    """Single inference metric record."""

    model_id: str
    version: str
    timestamp: float
    latency_ms: float
    success: bool
    error_type: str | None = None
    input_size: int | None = None
    output_size: int | None = None
    memory_usage_mb: float | None = None


@dataclass
class ModelHealthMetric:
    """Model health status metric."""

    model_id: str
    version: str
    timestamp: float
    status: str  # "healthy", "degraded", "unhealthy"
    uptime_seconds: float
    total_inferences: int
    success_rate: float
    avg_latency_ms: float
    memory_usage_mb: float
    error_count: int


class ModelMonitoringConfig(BaseModel):
    """Configuration for model monitoring."""

    enable_prometheus: bool = True
    prometheus_port: int = 8091

    # Metrics collection
    collect_inference_metrics: bool = True
    collect_health_metrics: bool = True
    collect_system_metrics: bool = True

    # Performance thresholds
    latency_threshold_ms: float = 1000.0
    error_rate_threshold: float = 0.05  # 5%
    memory_threshold_mb: float = 1024.0

    # Time windows
    metrics_window_minutes: int = 60
    health_check_interval_seconds: int = 30

    # Retention
    max_inference_records: int = 10000
    max_health_records: int = 1000

    # Alerting
    enable_alerting: bool = True
    alert_webhook_url: str | None = None


class ModelMonitoringService:
    """Comprehensive ML Model Monitoring Service.

    Features:
    - Real-time inference metrics collection
    - Model health monitoring and alerting
    - Prometheus metrics export
    - Performance trend analysis
    - Anomaly detection
    - Custom dashboards and reporting
    """

    def __init__(self, config: ModelMonitoringConfig | None = None) -> None:
        self.config = config or ModelMonitoringConfig()
        self.model_manager: ModelManager | None = None

        # Metrics storage
        self.inference_metrics: deque[ModelInferenceMetric] = deque(
            maxlen=self.config.max_inference_records
        )
        self.health_metrics: deque[ModelHealthMetric] = deque(
            maxlen=self.config.max_health_records
        )

        # Time-series data for trends
        self.latency_trends: defaultdict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self.error_trends: defaultdict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Prometheus metrics
        self.prometheus_metrics = self._setup_prometheus_metrics()

        # Monitoring tasks
        self.monitoring_tasks: list[asyncio.Task[None]] = []

        # Alert state tracking
        self.alert_state: dict[str, dict[str, Any]] = {}

        logger.info("Model monitoring service initialized")

    async def initialize(self, model_manager: ModelManager) -> None:
        """Initialize monitoring service with model manager."""
        self.model_manager = model_manager

        # Start Prometheus server if enabled
        if self.config.enable_prometheus:
            try:
                start_http_server(self.config.prometheus_port)
                logger.info(
                    "Prometheus metrics server started on port %s",
                    self.config.prometheus_port,
                )
            except Exception as e:
                logger.exception("Failed to start Prometheus server: %s", e)

        # Start monitoring tasks
        if self.config.collect_health_metrics:
            self.monitoring_tasks.append(
                asyncio.create_task(self._health_monitoring_loop())
            )

        if self.config.collect_system_metrics:
            self.monitoring_tasks.append(
                asyncio.create_task(self._system_monitoring_loop())
            )

        logger.info("Model monitoring service started")

    async def shutdown(self) -> None:
        """Shutdown monitoring service."""
        for task in self.monitoring_tasks:
            task.cancel()

        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        logger.info("Model monitoring service shutdown")

    def record_inference(
        self,
        model_id: str,
        version: str,
        latency_ms: float,
        *,
        success: bool,
        error_type: str | None = None,
        input_size: int | None = None,
        output_size: int | None = None,
        memory_usage_mb: float | None = None,
    ) -> None:
        """Record an inference event."""
        if not self.config.collect_inference_metrics:
            return

        metric = ModelInferenceMetric(
            model_id=model_id,
            version=version,
            timestamp=time.time(),
            latency_ms=latency_ms,
            success=success,
            error_type=error_type,
            input_size=input_size,
            output_size=output_size,
            memory_usage_mb=memory_usage_mb,
        )

        self.inference_metrics.append(metric)

        # Update Prometheus metrics
        model_key = f"{model_id}:{version}"

        if self.prometheus_metrics:
            self.prometheus_metrics["inference_count"].labels(
                model_id=model_id,
                version=version,
                status="success" if success else "error",
            ).inc()

            self.prometheus_metrics["inference_latency"].labels(
                model_id=model_id, version=version
            ).observe(
                latency_ms / 1000.0
            )  # Convert to seconds

            if memory_usage_mb:
                self.prometheus_metrics["memory_usage"].labels(
                    model_id=model_id, version=version
                ).set(memory_usage_mb)

        # Update trends
        self.latency_trends[model_key].append(latency_ms)
        self.error_trends[model_key].append(0.0 if success else 1.0)

        # Check for alerts
        self._alert_task = asyncio.create_task(
            self._check_inference_alerts(model_id, version, metric)
        )

    async def get_model_metrics(
        self, model_id: str, version: str = "latest", window_minutes: int | None = None
    ) -> dict[str, Any]:
        """Get comprehensive metrics for a specific model."""
        window_minutes = window_minutes or self.config.metrics_window_minutes
        window_start = time.time() - (window_minutes * 60)

        model_key = f"{model_id}:{version}"

        # Filter metrics within time window
        inference_metrics = [
            m
            for m in self.inference_metrics
            if m.model_id == model_id
            and m.version == version
            and m.timestamp >= window_start
        ]

        health_metrics = [
            m
            for m in self.health_metrics
            if m.model_id == model_id
            and m.version == version
            and m.timestamp >= window_start
        ]

        if not inference_metrics:
            return {"error": "No metrics found for model", "model": model_key}

        # Calculate aggregate metrics
        total_inferences = len(inference_metrics)
        successful_inferences = sum(1 for m in inference_metrics if m.success)
        failed_inferences = total_inferences - successful_inferences

        latencies = [m.latency_ms for m in inference_metrics]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0

        success_rate = (
            successful_inferences / total_inferences if total_inferences > 0 else 0
        )

        # Memory usage
        memory_usage = [
            m.memory_usage_mb for m in inference_metrics if m.memory_usage_mb
        ]
        avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0

        # Error breakdown
        error_types: defaultdict[str, int] = defaultdict(int)
        for m in inference_metrics:
            if not m.success and m.error_type:
                error_types[m.error_type] += 1

        # Trend analysis
        latency_trend = list(self.latency_trends[model_key])
        error_trend = list(self.error_trends[model_key])

        return {
            "model_id": model_id,
            "version": version,
            "window_minutes": window_minutes,
            "summary": {
                "total_inferences": total_inferences,
                "successful_inferences": successful_inferences,
                "failed_inferences": failed_inferences,
                "success_rate": success_rate,
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
                "p99_latency_ms": p99_latency,
                "avg_memory_mb": avg_memory,
            },
            "errors": dict(error_types),
            "trends": {"latency": latency_trend, "error_rate": error_trend},
            "health_checks": len(health_metrics),
            "alerts": self.alert_state.get(model_key, {}),
        }

    async def get_all_models_metrics(
        self, window_minutes: int | None = None
    ) -> dict[str, Any]:
        """Get metrics for all monitored models."""
        metrics = {}

        # Get unique model combinations
        model_combinations: set[tuple[str, str]] = set()
        model_combinations.update(
            (metric.model_id, metric.version) for metric in self.inference_metrics
        )

        for model_id, version in model_combinations:
            model_key = f"{model_id}:{version}"
            metrics[model_key] = await self.get_model_metrics(
                model_id, version, window_minutes
            )

        return metrics

    async def get_system_overview(self) -> dict[str, Any]:
        """Get system-wide monitoring overview."""
        if not self.model_manager:
            return {"error": "Model manager not available"}

        # Get model manager health
        manager_health = await self.model_manager.health_check()

        # Calculate system metrics
        total_inferences = len(self.inference_metrics)
        recent_inferences = sum(
            1
            for m in self.inference_metrics
            if m.timestamp >= time.time() - 300  # Last 5 minutes
        )

        success_rate = (
            sum(1 for m in self.inference_metrics if m.success) / total_inferences
            if total_inferences > 0
            else 0
        )

        # Active alerts
        active_alerts = sum(len(alerts) for alerts in self.alert_state.values())

        return {
            "system_status": manager_health.get("status", "unknown"),
            "loaded_models": manager_health.get("loaded_models", 0),
            "total_memory_mb": manager_health.get("total_memory_mb", 0),
            "metrics": {
                "total_inferences": total_inferences,
                "recent_inferences_5min": recent_inferences,
                "overall_success_rate": success_rate,
                "active_alerts": active_alerts,
            },
            "monitoring": {
                "prometheus_enabled": self.config.enable_prometheus,
                "prometheus_port": self.config.prometheus_port,
                "collection_enabled": self.config.collect_inference_metrics,
                "health_monitoring": self.config.collect_health_metrics,
            },
        }

    def _setup_prometheus_metrics(self) -> dict[str, Any] | None:
        """Setup Prometheus metrics."""
        if not self.config.enable_prometheus:
            return None

        try:
            return {
                "inference_count": Counter(
                    "model_inference_total",
                    "Total number of model inferences",
                    ["model_id", "version", "status"],
                ),
                "inference_latency": Histogram(
                    "model_inference_latency_seconds",
                    "Model inference latency in seconds",
                    ["model_id", "version"],
                    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                ),
                "memory_usage": Gauge(
                    "model_memory_usage_mb",
                    "Model memory usage in MB",
                    ["model_id", "version"],
                ),
                "model_health": Gauge(
                    "model_health_status",
                    "Model health status (1=healthy, 0=unhealthy)",
                    ["model_id", "version"],
                ),
                "system_info": Info("model_system_info", "Model system information"),
            }
        except Exception as e:
            logger.exception("Failed to setup Prometheus metrics: %s", e)
            return None

    async def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop."""
        while True:  # noqa: PLR1702 - Background monitoring loop
            try:
                if self.model_manager:
                    health_data = await self.model_manager.health_check()

                    # Record health metrics for each model
                    for model_id, model_data in health_data.get("models", {}).items():
                        model_parts = model_id.split(":", 1)
                        if len(model_parts) == MODEL_ID_PARTS:
                            model_name, version = model_parts

                            metric = ModelHealthMetric(
                                model_id=model_name,
                                version=version,
                                timestamp=time.time(),
                                status=(
                                    "healthy"
                                    if model_data.get("error_rate", 1.0)
                                    < self.config.error_rate_threshold
                                    else "unhealthy"
                                ),
                                uptime_seconds=model_data.get("uptime_seconds", 0),
                                total_inferences=model_data.get("total_inferences", 0),
                                success_rate=1.0 - model_data.get("error_rate", 0),
                                avg_latency_ms=model_data.get("avg_latency_ms", 0),
                                memory_usage_mb=model_data.get("memory_usage_mb", 0),
                                error_count=model_data.get("error_count", 0),
                            )

                            self.health_metrics.append(metric)

                            # Update Prometheus health metric
                            if self.prometheus_metrics:
                                health_value = (
                                    1.0 if metric.status == "healthy" else 0.0
                                )
                                self.prometheus_metrics["model_health"].labels(
                                    model_id=model_name, version=version
                                ).set(health_value)

                await asyncio.sleep(self.config.health_check_interval_seconds)

            except Exception as e:
                logger.exception("Health monitoring error: %s", e)
                await asyncio.sleep(self.config.health_check_interval_seconds)

    async def _system_monitoring_loop(self) -> None:
        """Background system monitoring loop."""
        while True:
            try:
                # Update system info in Prometheus
                if self.prometheus_metrics and self.model_manager:
                    system_info = await self.get_system_overview()

                    self.prometheus_metrics["system_info"].info(
                        {
                            "status": system_info.get("system_status", "unknown"),
                            "loaded_models": str(system_info.get("loaded_models", 0)),
                            "total_memory_mb": str(
                                system_info.get("total_memory_mb", 0)
                            ),
                        }
                    )

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.exception("System monitoring error: %s", e)
                await asyncio.sleep(60)

    async def _check_inference_alerts(
        self, model_id: str, version: str, metric: ModelInferenceMetric
    ) -> None:
        """Check for inference-related alerts."""
        if not self.config.enable_alerting:
            return

        model_key = f"{model_id}:{version}"

        # Initialize alert state if needed
        if model_key not in self.alert_state:
            self.alert_state[model_key] = {}

        alerts = []

        # Check latency threshold
        if metric.latency_ms > self.config.latency_threshold_ms:
            alerts.append(
                {
                    "type": "high_latency",
                    "severity": "warning",
                    "message": f"High latency detected: {metric.latency_ms:.2f}ms > {self.config.latency_threshold_ms}ms",
                    "timestamp": metric.timestamp,
                }
            )

        # Check memory threshold
        if (
            metric.memory_usage_mb
            and metric.memory_usage_mb > self.config.memory_threshold_mb
        ):
            alerts.append(
                {
                    "type": "high_memory",
                    "severity": "warning",
                    "message": f"High memory usage: {metric.memory_usage_mb:.2f}MB > {self.config.memory_threshold_mb}MB",
                    "timestamp": metric.timestamp,
                }
            )

        # Check error rate (over recent window)
        recent_errors = [
            m
            for m in self.inference_metrics
            if (
                m.model_id == model_id
                and m.version == version
                and m.timestamp >= time.time() - 300
            )  # Last 5 minutes
        ]

        if len(recent_errors) >= MIN_ERROR_SAMPLE_SIZE:
            error_rate = sum(1 for m in recent_errors if not m.success) / len(
                recent_errors
            )
            if error_rate > self.config.error_rate_threshold:
                alerts.append(
                    {
                        "type": "high_error_rate",
                        "severity": "critical",
                        "message": f"High error rate: {error_rate:.2%} > {self.config.error_rate_threshold:.2%}",
                        "timestamp": metric.timestamp,
                    }
                )

        # Update alert state
        for alert in alerts:
            alert_key = f"{alert['type']}_{alert['severity']}"
            self.alert_state[model_key][alert_key] = alert

            # Send webhook if configured
            if self.config.alert_webhook_url:
                self._webhook_task = asyncio.create_task(
                    self._send_alert_webhook(model_key, alert)
                )

    async def _send_alert_webhook(self, model_key: str, alert: dict[str, Any]) -> None:
        """Send alert to webhook."""
        try:
            import aiohttp  # noqa: PLC0415

            payload = {
                "model": model_key,
                "alert": alert,
                "timestamp": datetime.now(UTC).isoformat(),
                "service": "clarity-model-monitoring",
            }

            if not self.config.alert_webhook_url:
                return

            async with (
                aiohttp.ClientSession() as session,
                session.post(self.config.alert_webhook_url, json=payload) as response,
            ):
                if response.status != HTTP_OK:
                    logger.warning("Alert webhook failed: %s", response.status)
                else:
                    logger.info("Alert sent for %s: %s", model_key, alert["type"])

        except Exception as e:
            logger.exception("Failed to send alert webhook: %s", e)


# Decorator for automatic inference monitoring
T = TypeVar("T", bound=Callable[..., Awaitable[Any]])


def monitor_inference(
    monitoring_service: ModelMonitoringService,
) -> Callable[[T], T]:
    """Decorator to automatically monitor model inferences."""

    def decorator(func: T) -> T:
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            success = False
            error_type = None

            try:
                result = await func(self, *args, **kwargs)
                success = True
                return result
            except Exception as e:
                error_type = type(e).__name__
                raise
            finally:
                latency_ms = (time.time() - start_time) * 1000

                # Extract model info
                model_id = getattr(self, "model_id", "unknown")
                version = getattr(self, "version", "unknown")

                monitoring_service.record_inference(
                    model_id=model_id,
                    version=version,
                    latency_ms=latency_ms,
                    success=success,
                    error_type=error_type,
                )

        return wrapper  # type: ignore[return-value]

    return decorator
