"""Test secure logging module."""

from datetime import UTC, datetime
import logging
from unittest.mock import Mock
import uuid

import pytest

from clarity.core.secure_logging import (
    log_health_data_received,
    log_health_metrics_processed,
    sanitize_for_logging,
)
from clarity.models.health_data import HealthDataUpload, HealthMetric, HealthMetricType


def test_log_health_data_received(caplog):
    """Test logging health data reception."""
    logger = logging.getLogger("test")

    # Create mock health data
    mock_data = Mock(spec=HealthDataUpload)
    mock_data.user_id = "user123"
    mock_data.metrics = [Mock(), Mock(), Mock()]  # 3 metrics
    mock_data.upload_source = "apple_health"

    with caplog.at_level(logging.INFO):
        log_health_data_received(logger, mock_data)

    assert "Received health data for user user123" in caplog.text
    assert "3 metrics" in caplog.text
    assert "source: apple_health" in caplog.text


def test_log_health_metrics_processed(caplog):
    """Test logging health metrics processing."""
    logger = logging.getLogger("test")

    # Create mock metrics
    metric1 = Mock(spec=HealthMetric)
    metric1.metric_type = Mock(value="heart_rate")

    metric2 = Mock(spec=HealthMetric)
    metric2.metric_type = Mock(value="sleep_analysis")

    metric3 = Mock(spec=HealthMetric)
    metric3.metric_type = Mock(value="heart_rate")

    metrics = [metric1, metric2, metric3]

    with caplog.at_level(logging.INFO):
        log_health_metrics_processed(logger, "user456", metrics)

    assert "Processed 3 health metrics for user user456" in caplog.text
    assert "heart_rate" in caplog.text
    assert "sleep_analysis" in caplog.text


def test_sanitize_for_logging_health_data():
    """Test sanitizing health data objects."""
    mock_data = Mock()
    mock_data.user_id = "user789"
    mock_data.metrics = [1, 2, 3, 4, 5]

    result = sanitize_for_logging(mock_data)
    assert result == "HealthData(user_id=user789, metrics_count=5)"


def test_sanitize_for_logging_user_data():
    """Test sanitizing user data objects."""
    mock_data = Mock()
    mock_data.user_id = "user999"
    # Ensure the object doesn't have a metrics attribute
    del mock_data.metrics

    result = sanitize_for_logging(mock_data)
    assert result == "Data(user_id=user999)"


def test_sanitize_for_logging_dict():
    """Test sanitizing dictionary with sensitive keys."""
    data = {
        "user_id": "user123",
        "email": "test@example.com",
        "phone": "555-1234",
        "address": "123 Main St",
        "ssn": "123-45-6789",
        "dob": "1990-01-01",
        "safe_field": "visible_value",
    }

    result = sanitize_for_logging(data)
    assert "[MASKED]" in result
    assert "visible_value" in result
    assert "test@example.com" not in result
    assert "555-1234" not in result


def test_sanitize_for_logging_dict_with_complex_values():
    """Test sanitizing dictionary with large complex values."""
    large_list = list(range(1000))
    large_dict = {f"key{i}": i for i in range(100)}

    data = {"simple": "value", "large_list": large_list, "large_dict": large_dict}

    result = sanitize_for_logging(data)
    assert "[list:1000]" in result
    assert "[dict:100]" in result
    assert "value" in result


def test_sanitize_for_logging_other_types():
    """Test sanitizing other data types."""
    assert sanitize_for_logging("simple string") == "simple string"
    assert sanitize_for_logging(123) == "123"
    assert sanitize_for_logging([1, 2, 3]) == "[1, 2, 3]"
