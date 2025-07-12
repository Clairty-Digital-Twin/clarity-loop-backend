"""PAT Model Observability Metrics.

Comprehensive monitoring for PAT model operations including:
- Model loading attempts and failures
- Checksum verification tracking
- Performance metrics
- Security violation alerts

This module provides Prometheus metrics for production monitoring
and alerting of the PAT (Proxy Actigraphy Transformer) model system.
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

from prometheus_client import Counter, Gauge, Histogram, Summary

logger = logging.getLogger(__name__)

# Model Loading Metrics
pat_model_load_attempts = Counter(
    "clarity_pat_model_load_attempts_total",
    "Total number of PAT model load attempts",
    ["model_size", "version", "source"],  # source: local, s3, cache
)

pat_model_load_failures = Counter(
    "clarity_pat_model_load_failures_total",
    "Total number of PAT model load failures",
    ["model_size", "version", "error_type"],
)

pat_model_load_duration = Histogram(
    "clarity_pat_model_load_duration_seconds",
    "Time taken to load PAT model in seconds",
    ["model_size", "version", "source"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60),  # Common load time buckets
)

# Checksum Verification Metrics
pat_checksum_verification_attempts = Counter(
    "clarity_pat_checksum_verification_attempts_total",
    "Total number of checksum verification attempts",
    ["model_size", "version"],
)

pat_checksum_verification_failures = Counter(
    "clarity_pat_checksum_verification_failures_total",
    "Total number of checksum verification failures (security violations)",
    ["model_size", "version", "expected_checksum"],
)

pat_checksum_verification_duration = Histogram(
    "clarity_pat_checksum_verification_duration_seconds",
    "Time taken to verify model checksum in seconds",
    ["model_size"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2),  # Checksum should be fast
)

# Model Cache Metrics
pat_model_cache_hits = Counter(
    "clarity_pat_model_cache_hits_total",
    "Total number of model cache hits",
    ["model_size", "version"],
)

pat_model_cache_misses = Counter(
    "clarity_pat_model_cache_misses_total",
    "Total number of model cache misses",
    ["model_size", "version"],
)

pat_model_cache_size = Gauge(
    "clarity_pat_model_cache_size_count",
    "Current number of models in cache",
)

pat_model_cache_memory_bytes = Gauge(
    "clarity_pat_model_cache_memory_bytes",
    "Estimated memory usage of model cache in bytes",
)

# Model Loading Progress
pat_model_loading_progress = Gauge(
    "clarity_pat_model_loading_progress_ratio",
    "Current model loading progress (0.0 to 1.0)",
    ["model_size", "version", "stage"],  # stage: download, checksum, load, validate
)

# Model Version Info
pat_model_current_version = Gauge(
    "clarity_pat_model_current_version_info",
    "Information about currently loaded model versions",
    ["model_size", "version", "checksum"],
)

# S3 Download Metrics
pat_model_s3_downloads = Counter(
    "clarity_pat_model_s3_downloads_total",
    "Total number of model downloads from S3",
    ["model_size", "version", "status"],
)

pat_model_s3_download_duration = Histogram(
    "clarity_pat_model_s3_download_duration_seconds",
    "Time taken to download model from S3 in seconds",
    ["model_size"],
    buckets=(1, 5, 10, 30, 60, 120, 300),  # S3 download can be slow
)

pat_model_s3_download_bytes = Summary(
    "clarity_pat_model_s3_download_bytes",
    "Size of model files downloaded from S3",
    ["model_size"],
)

# Model Validation Metrics
pat_model_validation_attempts = Counter(
    "clarity_pat_model_validation_attempts_total",
    "Total number of model validation attempts",
    ["model_size", "validation_type"],  # validation_type: shape, forward_pass
)

pat_model_validation_failures = Counter(
    "clarity_pat_model_validation_failures_total",
    "Total number of model validation failures",
    ["model_size", "validation_type", "error_type"],
)

# Fallback Metrics
pat_model_fallback_attempts = Counter(
    "clarity_pat_model_fallback_attempts_total",
    "Total number of model fallback attempts",
    ["model_size", "from_version", "to_version"],
)

pat_model_fallback_successes = Counter(
    "clarity_pat_model_fallback_successes_total",
    "Total number of successful model fallbacks",
    ["model_size", "from_version", "to_version"],
)

# Hot Swap Metrics
pat_model_hot_swap_attempts = Counter(
    "clarity_pat_model_hot_swap_attempts_total",
    "Total number of model hot swap attempts",
    ["model_size", "from_version", "to_version"],
)

pat_model_hot_swap_duration = Histogram(
    "clarity_pat_model_hot_swap_duration_seconds",
    "Time taken to perform model hot swap in seconds",
    ["model_size"],
    buckets=(0.1, 0.5, 1, 2, 5, 10),
)

# Security Alert Metrics
pat_security_violations = Counter(
    "clarity_pat_security_violations_total",
    "Total number of security violations detected",
    ["violation_type", "model_size", "severity"],  # severity: critical, high, medium
)

# Overall Model Health
pat_model_health_score = Gauge(
    "clarity_pat_model_health_score",
    "Overall health score of PAT model system (0-100)",
)


# Helper Functions
@contextmanager
def track_model_load(
    model_size: str, version: str, source: str
) -> Iterator[dict[str, Any]]:
    """Context manager to track model loading operations.
    
    Args:
        model_size: Size of the model (small, medium, large)
        version: Model version being loaded
        source: Source of the model (local, s3, cache)
        
    Yields:
        Context dict for storing operation results
    """
    start_time = time.time()
    context: dict[str, Any] = {"success": False, "error_type": None}
    
    pat_model_load_attempts.labels(
        model_size=model_size, version=version, source=source
    ).inc()
    
    try:
        yield context
    except Exception as e:
        context["success"] = False
        context["error_type"] = type(e).__name__
        raise
    finally:
        duration = time.time() - start_time
        
        if context["success"]:
            pat_model_load_duration.labels(
                model_size=model_size, version=version, source=source
            ).observe(duration)
        else:
            error_type = context.get("error_type", "unknown")
            pat_model_load_failures.labels(
                model_size=model_size, version=version, error_type=error_type
            ).inc()


@contextmanager
def track_checksum_verification(
    model_size: str, version: str
) -> Iterator[dict[str, Any]]:
    """Context manager to track checksum verification.
    
    Args:
        model_size: Size of the model
        version: Model version being verified
        
    Yields:
        Context dict for storing verification results
    """
    start_time = time.time()
    context: dict[str, Any] = {
        "success": False,
        "expected_checksum": None,
        "actual_checksum": None,
    }
    
    pat_checksum_verification_attempts.labels(
        model_size=model_size, version=version
    ).inc()
    
    try:
        yield context
    except Exception:
        context["success"] = False
        raise
    finally:
        duration = time.time() - start_time
        
        pat_checksum_verification_duration.labels(model_size=model_size).observe(
            duration
        )
        
        if not context["success"] and context.get("expected_checksum"):
            # Record checksum failure as security violation
            pat_checksum_verification_failures.labels(
                model_size=model_size,
                version=version,
                expected_checksum=context["expected_checksum"],
            ).inc()
            
            # Also record as security violation
            record_security_violation(
                "checksum_mismatch", model_size, "critical"
            )


def update_loading_progress(
    model_size: str, version: str, stage: str, progress: float
) -> None:
    """Update model loading progress.
    
    Args:
        model_size: Size of the model
        version: Model version being loaded
        stage: Current loading stage (download, checksum, load, validate)
        progress: Progress ratio (0.0 to 1.0)
    """
    pat_model_loading_progress.labels(
        model_size=model_size, version=version, stage=stage
    ).set(progress)


def record_cache_hit(model_size: str, version: str) -> None:
    """Record a model cache hit."""
    pat_model_cache_hits.labels(model_size=model_size, version=version).inc()


def record_cache_miss(model_size: str, version: str) -> None:
    """Record a model cache miss."""
    pat_model_cache_misses.labels(model_size=model_size, version=version).inc()


def update_cache_metrics(cache_size: int, memory_bytes: int) -> None:
    """Update cache size metrics.
    
    Args:
        cache_size: Number of models in cache
        memory_bytes: Estimated memory usage in bytes
    """
    pat_model_cache_size.set(cache_size)
    pat_model_cache_memory_bytes.set(memory_bytes)


def record_s3_download(
    model_size: str, version: str, status: str, duration: float | None = None,
    bytes_downloaded: int | None = None
) -> None:
    """Record S3 download metrics.
    
    Args:
        model_size: Size of the model
        version: Model version
        status: Download status (success, failed)
        duration: Download duration in seconds
        bytes_downloaded: Number of bytes downloaded
    """
    pat_model_s3_downloads.labels(
        model_size=model_size, version=version, status=status
    ).inc()
    
    if duration is not None:
        pat_model_s3_download_duration.labels(model_size=model_size).observe(duration)
    
    if bytes_downloaded is not None:
        pat_model_s3_download_bytes.labels(model_size=model_size).observe(
            bytes_downloaded
        )


def record_validation_attempt(
    model_size: str, validation_type: str, success: bool, error_type: str | None = None
) -> None:
    """Record model validation attempt.
    
    Args:
        model_size: Size of the model
        validation_type: Type of validation (shape, forward_pass)
        success: Whether validation succeeded
        error_type: Type of error if failed
    """
    pat_model_validation_attempts.labels(
        model_size=model_size, validation_type=validation_type
    ).inc()
    
    if not success and error_type:
        pat_model_validation_failures.labels(
            model_size=model_size,
            validation_type=validation_type,
            error_type=error_type,
        ).inc()


def record_fallback_attempt(
    model_size: str, from_version: str, to_version: str, success: bool
) -> None:
    """Record model fallback attempt.
    
    Args:
        model_size: Size of the model
        from_version: Version falling back from
        to_version: Version falling back to
        success: Whether fallback succeeded
    """
    pat_model_fallback_attempts.labels(
        model_size=model_size, from_version=from_version, to_version=to_version
    ).inc()
    
    if success:
        pat_model_fallback_successes.labels(
            model_size=model_size, from_version=from_version, to_version=to_version
        ).inc()


def record_hot_swap(
    model_size: str, from_version: str, to_version: str, duration: float
) -> None:
    """Record model hot swap operation.
    
    Args:
        model_size: Size of the model
        from_version: Version swapping from
        to_version: Version swapping to
        duration: Swap duration in seconds
    """
    pat_model_hot_swap_attempts.labels(
        model_size=model_size, from_version=from_version, to_version=to_version
    ).inc()
    
    pat_model_hot_swap_duration.labels(model_size=model_size).observe(duration)


def record_security_violation(
    violation_type: str, model_size: str, severity: str
) -> None:
    """Record a security violation.
    
    Args:
        violation_type: Type of violation (checksum_mismatch, unauthorized_access, etc.)
        model_size: Size of the model involved
        severity: Severity level (critical, high, medium)
    """
    pat_security_violations.labels(
        violation_type=violation_type, model_size=model_size, severity=severity
    ).inc()
    
    logger.error(
        "SECURITY VIOLATION: type=%s, model_size=%s, severity=%s",
        violation_type,
        model_size,
        severity,
    )


def update_current_version(model_size: str, version: str, checksum: str) -> None:
    """Update current model version info.
    
    Args:
        model_size: Size of the model
        version: Current version
        checksum: Model checksum
    """
    # Reset all labels for this model size first
    for label_values in pat_model_current_version._metrics.copy():
        if label_values[0] == model_size:
            pat_model_current_version.remove(*label_values)
    
    # Set new current version
    pat_model_current_version.labels(
        model_size=model_size, version=version, checksum=checksum
    ).set(1)


def calculate_model_health_score() -> float:
    """Calculate overall PAT model system health score.
    
    Returns:
        Health score from 0-100
    """
    # This is a simplified health score calculation
    # In production, this would incorporate multiple factors
    
    # Get recent metrics (this is pseudo-code, actual implementation would query metrics)
    health_score = 100.0
    
    # Deduct points for failures
    # - Each load failure: -5 points
    # - Each checksum failure: -20 points (security issue)
    # - Each validation failure: -10 points
    
    # In a real implementation, you would query the actual metric values
    # For now, return a static healthy score
    return health_score


def update_health_score() -> None:
    """Update the overall model health score metric."""
    score = calculate_model_health_score()
    pat_model_health_score.set(score)


# Export all metrics and helper functions
__all__ = [
    # Metrics
    "pat_model_load_attempts",
    "pat_model_load_failures",
    "pat_model_load_duration",
    "pat_checksum_verification_attempts",
    "pat_checksum_verification_failures",
    "pat_checksum_verification_duration",
    "pat_model_cache_hits",
    "pat_model_cache_misses",
    "pat_model_cache_size",
    "pat_model_cache_memory_bytes",
    "pat_model_loading_progress",
    "pat_model_current_version",
    "pat_model_s3_downloads",
    "pat_model_s3_download_duration",
    "pat_model_s3_download_bytes",
    "pat_model_validation_attempts",
    "pat_model_validation_failures",
    "pat_model_fallback_attempts",
    "pat_model_fallback_successes",
    "pat_model_hot_swap_attempts",
    "pat_model_hot_swap_duration",
    "pat_security_violations",
    "pat_model_health_score",
    # Helper functions
    "track_model_load",
    "track_checksum_verification",
    "update_loading_progress",
    "record_cache_hit",
    "record_cache_miss",
    "update_cache_metrics",
    "record_s3_download",
    "record_validation_attempt",
    "record_fallback_attempt",
    "record_hot_swap",
    "record_security_violation",
    "update_current_version",
    "update_health_score",
]