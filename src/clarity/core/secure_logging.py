"""HIPAA-compliant secure logging utilities.

Provides logging functions that automatically sanitize health data
to prevent accidental PHI exposure in logs.
"""

# removed - breaks FastAPI

import logging
from typing import TYPE_CHECKING

from clarity.models.health_data import HealthDataUpload, HealthMetric

if TYPE_CHECKING:
    pass  # Only for type stubs now

# Constants for sanitization
MAX_COMPLEX_VALUE_LENGTH = 100


def log_health_data_received(
    logger: logging.Logger, health_data: HealthDataUpload
) -> None:
    """Log health data reception without exposing PHI.

    HIPAA-compliant logging that only includes non-sensitive metadata.

    Args:
        logger: Logger instance to use
        health_data: Health data upload (PHI will NOT be logged)
    """
    logger.info(
        "Received health data for user %s (%d metrics, source: %s)",
        health_data.user_id,
        len(health_data.metrics),
        health_data.upload_source,
    )


def log_health_metrics_processed(
    logger: logging.Logger, user_id: str, metrics: list[HealthMetric]
) -> None:
    """Log health metrics processing without exposing PHI.

    Args:
        logger: Logger instance to use
        user_id: User identifier
        metrics: List of health metrics (PHI will NOT be logged)
    """
    metric_types = [m.metric_type.value for m in metrics]
    logger.info(
        "Processed %d health metrics for user %s (types: %s)",
        len(metrics),
        user_id,
        ", ".join(set(metric_types)),
    )


def sanitize_for_logging(data: object) -> str:
    """Sanitize data for safe logging.

    Removes or masks PHI from data structures before logging.

    Args:
        data: Data to sanitize

    Returns:
        Safe string representation for logging
    """
    if hasattr(data, "user_id"):
        # For health data objects, only log metadata
        if hasattr(data, "metrics"):
            return (
                f"HealthData(user_id={data.user_id}, metrics_count={len(data.metrics)})"
            )
        return f"Data(user_id={data.user_id})"

    if isinstance(data, dict):
        # For dictionaries, mask sensitive keys
        safe_dict = {}
        for key, value in data.items():
            if key.lower() in {"email", "phone", "address", "ssn", "dob"}:
                safe_dict[key] = "[MASKED]"
            elif (
                isinstance(value, (dict, list))
                and len(str(value)) > MAX_COMPLEX_VALUE_LENGTH
            ):
                safe_dict[key] = (
                    f"[{type(value).__name__}:{len(value) if hasattr(value, '__len__') else 'N/A'}]"
                )
            else:
                safe_dict[key] = value
        return str(safe_dict)

    # For other types, just return string representation
    return str(data)
