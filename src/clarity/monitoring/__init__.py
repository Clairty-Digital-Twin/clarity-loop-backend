"""Clarity monitoring module for observability and metrics."""

from clarity.monitoring.pat_metrics import (
    record_cache_hit,
    record_cache_miss,
    record_fallback_attempt,
    record_hot_swap,
    record_s3_download,
    record_security_violation,
    record_validation_attempt,
    track_checksum_verification,
    track_model_load,
    update_cache_metrics,
    update_current_version,
    update_health_score,
    update_loading_progress,
)

__all__ = [
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