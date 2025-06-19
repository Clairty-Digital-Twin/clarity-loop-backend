"""Prometheus Metrics API Endpoint.

Provides /metrics endpoint for Prometheus monitoring integration.
Exposes custom application metrics including:
- Health data processing metrics
- PAT model inference metrics
- Insight generation metrics
- API endpoint performance metrics
- System health metrics

Designed for production monitoring and alerting.
"""

# removed - breaks FastAPI

import logging
import time
from typing import TYPE_CHECKING, Any, Self

try:
    import psutil  # type: ignore[import-untyped]
except ImportError:
    psutil = None


import types

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

if TYPE_CHECKING:
    pass  # Only for type stubs now

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="", tags=["metrics"])

# 🔥 Prometheus Metrics Definitions

# Request metrics
http_requests_total = Counter(
    "clarity_http_requests_total",
    "Total HTTP requests processed",
    ["method", "endpoint", "status_code"],
)

http_request_duration_seconds = Histogram(
    "clarity_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

# Health data processing metrics
health_data_uploads_total = Counter(
    "clarity_health_data_uploads_total",
    "Total health data uploads processed",
    ["status", "source"],
)

health_data_processing_duration_seconds = Histogram(
    "clarity_health_data_processing_duration_seconds",
    "Health data processing time in seconds",
    ["stage"],
)

health_metrics_processed_total = Counter(
    "clarity_health_metrics_processed_total",
    "Total individual health metrics processed",
    ["metric_type"],
)

# PAT model metrics
pat_inference_requests_total = Counter(
    "clarity_pat_inference_requests_total",
    "Total PAT model inference requests",
    ["status"],
)

pat_inference_duration_seconds = Histogram(
    "clarity_pat_inference_duration_seconds", "PAT model inference duration in seconds"
)

pat_model_loading_time_seconds = Gauge(
    "clarity_pat_model_loading_time_seconds", "Time taken to load PAT model weights"
)

# Insight generation metrics
insight_generation_requests_total = Counter(
    "clarity_insight_generation_requests_total",
    "Total insight generation requests",
    ["status", "model"],
)

insight_generation_duration_seconds = Histogram(
    "clarity_insight_generation_duration_seconds",
    "Insight generation duration in seconds",
    ["model"],
)

# System health metrics
system_memory_usage_bytes = Gauge(
    "clarity_system_memory_usage_bytes", "Current system memory usage in bytes"
)

active_processing_jobs = Gauge(
    "clarity_active_processing_jobs", "Number of currently active processing jobs"
)

failed_jobs_total = Counter(
    "clarity_failed_jobs_total",
    "Total number of failed processing jobs",
    ["job_type", "error_type"],
)

# Database metrics
dynamodb_operations_total = Counter(
    "clarity_dynamodb_operations_total",
    "Total DynamoDB operations",
    ["operation", "collection", "status"],
)

dynamodb_operation_duration_seconds = Histogram(
    "clarity_dynamodb_operation_duration_seconds",
    "DynamoDB operation duration in seconds",
    ["operation", "collection"],
)

# Pub/Sub metrics
pubsub_messages_total = Counter(
    "clarity_pubsub_messages_total", "Total Pub/Sub messages", ["topic", "status"]
)

pubsub_message_processing_duration_seconds = Histogram(
    "clarity_pubsub_message_processing_duration_seconds",
    "Pub/Sub message processing duration in seconds",
    ["topic"],
)


@router.get(
    "/metrics",
    summary="Prometheus Metrics",
    description="Expose application metrics for Prometheus scraping",
    response_class=Response,
)
async def get_metrics() -> Response:
    """🔥 FIXED: Prometheus metrics endpoint for production monitoring.

    Returns:
        Response with Prometheus-formatted metrics data
    """
    try:
        logger.debug("Generating Prometheus metrics")

        # Update system metrics before generating output
        _update_system_metrics()

        # Generate Prometheus metrics format
        metrics_data = generate_latest()

        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    except Exception:
        logger.exception("Failed to generate metrics")
        # Return empty metrics rather than failing
        return Response(
            content=(
                "# HELP clarity_metrics_error Metrics generation error\n"
                "# TYPE clarity_metrics_error gauge\n"
                "clarity_metrics_error 1\n"
            ),
            media_type=CONTENT_TYPE_LATEST,
        )


def _update_system_metrics() -> None:
    """Update system-level metrics before exposition."""
    try:
        if psutil is None:
            # psutil not available - skip system metrics
            logger.debug("psutil not available for system metrics")
            return

        # Memory usage
        memory = psutil.virtual_memory()
        system_memory_usage_bytes.set(memory.used)

    except Exception:
        logger.exception("Failed to update system metrics")


# 🔥 Metric Helper Functions


def record_http_request(
    method: str, endpoint: str, status_code: int, duration: float
) -> None:
    """Record HTTP request metrics.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint path
        status_code: HTTP status code
        duration: Request duration in seconds
    """
    http_requests_total.labels(
        method=method, endpoint=endpoint, status_code=str(status_code)
    ).inc()

    http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(
        duration
    )


def record_health_data_upload(status: str, source: str) -> None:
    """Record health data upload metrics.

    Args:
        status: Upload status (success, failed, etc.)
        source: Data source (apple_health, manual, etc.)
    """
    health_data_uploads_total.labels(status=status, source=source).inc()


def record_health_data_processing(stage: str, duration: float) -> None:
    """Record health data processing metrics.

    Args:
        stage: Processing stage (preprocessing, analysis, etc.)
        duration: Processing duration in seconds
    """
    health_data_processing_duration_seconds.labels(stage=stage).observe(duration)


def record_health_metric_processed(metric_type: str) -> None:
    """Record individual health metric processing.

    Args:
        metric_type: Type of health metric (heart_rate, steps, etc.)
    """
    health_metrics_processed_total.labels(metric_type=metric_type).inc()


def record_pat_inference(status: str, duration: float | None = None) -> None:
    """Record PAT model inference metrics.

    Args:
        status: Inference status (success, failed, etc.)
        duration: Inference duration in seconds (optional)
    """
    pat_inference_requests_total.labels(status=status).inc()

    if duration is not None:
        pat_inference_duration_seconds.observe(duration)


def record_pat_model_loading(duration: float) -> None:
    """Record PAT model loading time.

    Args:
        duration: Model loading duration in seconds
    """
    pat_model_loading_time_seconds.set(duration)


def record_insight_generation(
    status: str, model: str, duration: float | None = None
) -> None:
    """Record insight generation metrics.

    Args:
        status: Generation status (success, failed, etc.)
        model: Model used (gemini-2.0-flash-exp, etc.)
        duration: Generation duration in seconds (optional)
    """
    insight_generation_requests_total.labels(status=status, model=model).inc()

    if duration is not None:
        insight_generation_duration_seconds.labels(model=model).observe(duration)


def record_processing_job_status(active_count: int) -> None:
    """Record active processing job count.

    Args:
        active_count: Number of currently active jobs
    """
    active_processing_jobs.set(active_count)


def record_failed_job(job_type: str, error_type: str) -> None:
    """Record failed job metrics.

    Args:
        job_type: Type of job (analysis, insight, etc.)
        error_type: Type of error (timeout, validation, etc.)
    """
    failed_jobs_total.labels(job_type=job_type, error_type=error_type).inc()


def record_dynamodb_operation(
    operation: str, table: str, status: str, duration: float | None = None
) -> None:
    """Record DynamoDB operation metrics.

    Args:
        operation: Operation type (create, read, update, delete)
        table: DynamoDB table name
        status: Operation status (success, failed, etc.)
        duration: Operation duration in seconds (optional)
    """
    dynamodb_operations_total.labels(
        operation=operation, collection=table, status=status
    ).inc()

    if duration is not None:
        dynamodb_operation_duration_seconds.labels(
            operation=operation, collection=table
        ).observe(duration)


def record_pubsub_message(
    topic: str, status: str, processing_duration: float | None = None
) -> None:
    """Record Pub/Sub message metrics.

    Args:
        topic: Pub/Sub topic name
        status: Message status (success, failed, etc.)
        processing_duration: Message processing duration in seconds (optional)
    """
    pubsub_messages_total.labels(topic=topic, status=status).inc()

    if processing_duration is not None:
        pubsub_message_processing_duration_seconds.labels(topic=topic).observe(
            processing_duration
        )


# 🔥 Metrics Middleware Integration Helper


class MetricsContext:
    """Context manager for automatic metrics recording."""

    def __init__(
        self, operation_type: str, labels: dict[str, Any] | None = None
    ) -> None:
        self.operation_type = operation_type
        self.labels = labels or {}
        self.start_time = time.time()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        duration = time.time() - self.start_time

        status = "success" if exc_type is None else "failed"

        # Record based on operation type
        if self.operation_type == "pat_inference":
            record_pat_inference(status, duration)
        elif self.operation_type == "insight_generation":
            model = self.labels.get("model", "unknown")
            record_insight_generation(status, model, duration)
        elif self.operation_type == "health_data_processing":
            stage = self.labels.get("stage", "unknown")
            record_health_data_processing(stage, duration)
        elif self.operation_type == "dynamodb_operation":
            operation = self.labels.get("operation", "unknown")
            table = self.labels.get("table", "unknown")
            record_dynamodb_operation(operation, table, status, duration)


# Export router and helper functions
__all__ = [
    "MetricsContext",
    "record_dynamodb_operation",
    "record_failed_job",
    "record_health_data_processing",
    "record_health_data_upload",
    "record_health_metric_processed",
    "record_http_request",
    "record_insight_generation",
    "record_pat_inference",
    "record_pat_model_loading",
    "record_processing_job_status",
    "record_pubsub_message",
    "router",
]
