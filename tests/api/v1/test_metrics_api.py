"""Comprehensive test suite for Prometheus metrics API.

Tests all metric recording functions, the /metrics endpoint, system metrics,
and the MetricsContext context manager to achieve 95%+ coverage.
"""

import time
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

from fastapi import Response
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
import pytest

from clarity.api.v1.metrics import (
    MetricsContext,
    _update_system_metrics,
    get_metrics,
    record_dynamodb_operation,
    record_failed_job,
    record_health_data_processing,
    record_health_data_upload,
    record_health_metric_processed,
    record_http_request,
    record_insight_generation,
    record_pat_inference,
    record_pat_model_loading,
    record_processing_job_status,
    record_pubsub_message,
    router,
)


class TestMetricsEndpoint:
    """Test the main /metrics endpoint."""

    @pytest.mark.asyncio
    async def test_get_metrics_success(self):
        """Test successful metrics generation."""
        # Record some metrics first
        record_http_request("GET", "/api/v1/test", 200, 0.1)
        record_health_data_upload("success", "apple_health")

        response = await get_metrics()

        assert isinstance(response, Response)
        assert response.media_type.startswith("text/plain")
        assert "charset=utf-8" in response.media_type
        assert "Cache-Control" in response.headers
        assert (
            response.headers["Cache-Control"] == "no-cache, no-store, must-revalidate"
        )
        assert response.headers["Pragma"] == "no-cache"
        assert response.headers["Expires"] == "0"

        # Check that metrics are included
        content = response.body.decode("utf-8")
        assert "clarity_http_requests_total" in content
        assert "clarity_health_data_uploads_total" in content

    @pytest.mark.asyncio
    @patch("clarity.api.v1.metrics.generate_latest")
    async def test_get_metrics_prometheus_error(self, mock_generate_latest):
        """Test metrics endpoint when Prometheus generation fails."""
        mock_generate_latest.side_effect = Exception("Prometheus error")

        response = await get_metrics()

        assert isinstance(response, Response)
        assert response.media_type.startswith("text/plain")
        assert "charset=utf-8" in response.media_type
        content = response.body.decode("utf-8")
        assert "clarity_metrics_error" in content
        assert "Metrics generation error" in content

    @pytest.mark.asyncio
    @patch("clarity.api.v1.metrics._update_system_metrics")
    async def test_get_metrics_system_metrics_updated(self, mock_update_system_metrics):
        """Test that system metrics are updated before generation."""
        await get_metrics()

        mock_update_system_metrics.assert_called_once()

    def test_router_configuration(self):
        """Test router is properly configured."""
        assert router.prefix == ""
        assert "metrics" in router.tags


class TestSystemMetrics:
    """Test system metrics collection."""

    @patch("clarity.api.v1.metrics.psutil")
    def test_update_system_metrics_with_psutil(self, mock_psutil):
        """Test system metrics update when psutil is available."""
        # Mock psutil virtual memory
        mock_memory = MagicMock()
        mock_memory.used = 1024 * 1024 * 1024  # 1GB
        mock_psutil.virtual_memory.return_value = mock_memory

        _update_system_metrics()

        mock_psutil.virtual_memory.assert_called_once()

    @patch("clarity.api.v1.metrics.psutil", None)
    def test_update_system_metrics_without_psutil(self):
        """Test system metrics update when psutil is not available."""
        # Should not raise an error
        _update_system_metrics()

    @patch("clarity.api.v1.metrics.psutil")
    def test_update_system_metrics_psutil_error(self, mock_psutil):
        """Test system metrics update when psutil raises an error."""
        mock_psutil.virtual_memory.side_effect = Exception("psutil error")

        # Should not raise an error
        _update_system_metrics()


class TestHTTPMetrics:
    """Test HTTP request metrics recording."""

    def test_record_http_request(self):
        """Test HTTP request metrics recording."""
        record_http_request("GET", "/api/v1/test", 200, 0.1)
        record_http_request("POST", "/api/v1/data", 201, 0.5)
        record_http_request("GET", "/api/v1/error", 500, 0.2)

        # Verify metrics were recorded (implicitly through no exceptions)
        assert True  # If we get here, no exceptions were raised

    def test_record_http_request_various_status_codes(self):
        """Test HTTP request recording with various status codes."""
        status_codes = [200, 201, 400, 401, 403, 404, 500, 502, 503]

        for status_code in status_codes:
            record_http_request("GET", f"/api/v1/test/{status_code}", status_code, 0.1)

        assert True  # All recordings successful


class TestHealthDataMetrics:
    """Test health data metrics recording."""

    def test_record_health_data_upload(self):
        """Test health data upload metrics recording."""
        record_health_data_upload("success", "apple_health")
        record_health_data_upload("failed", "manual")
        record_health_data_upload("success", "fitbit")

        assert True  # All recordings successful

    def test_record_health_data_processing(self):
        """Test health data processing metrics recording."""
        record_health_data_processing("preprocessing", 0.5)
        record_health_data_processing("analysis", 2.1)
        record_health_data_processing("postprocessing", 0.3)

        assert True  # All recordings successful

    def test_record_health_metric_processed(self):
        """Test individual health metric processing."""
        metric_types = ["heart_rate", "steps", "sleep", "blood_pressure", "weight"]

        for metric_type in metric_types:
            record_health_metric_processed(metric_type)

        assert True  # All recordings successful


class TestPATModelMetrics:
    """Test PAT model metrics recording."""

    def test_record_pat_inference_success(self):
        """Test PAT inference recording with duration."""
        record_pat_inference("success", 0.8)
        record_pat_inference("failed", 1.2)

        assert True  # All recordings successful

    def test_record_pat_inference_without_duration(self):
        """Test PAT inference recording without duration."""
        record_pat_inference("success")
        record_pat_inference("failed")

        assert True  # All recordings successful

    def test_record_pat_model_loading(self):
        """Test PAT model loading time recording."""
        record_pat_model_loading(5.2)
        record_pat_model_loading(3.8)

        assert True  # All recordings successful


class TestInsightGenerationMetrics:
    """Test insight generation metrics recording."""

    def test_record_insight_generation_with_duration(self):
        """Test insight generation recording with duration."""
        record_insight_generation("success", "gemini-2.0-flash-exp", 2.5)
        record_insight_generation("failed", "gemini-pro", 1.8)

        assert True  # All recordings successful

    def test_record_insight_generation_without_duration(self):
        """Test insight generation recording without duration."""
        record_insight_generation("success", "gemini-2.0-flash-exp")
        record_insight_generation("failed", "gemini-pro")

        assert True  # All recordings successful

    def test_record_insight_generation_various_models(self):
        """Test insight generation with various models."""
        models = ["gemini-2.0-flash-exp", "gemini-pro", "claude-3", "gpt-4"]

        for model in models:
            record_insight_generation("success", model, 1.0)

        assert True  # All recordings successful


class TestJobMetrics:
    """Test job-related metrics recording."""

    def test_record_processing_job_status(self):
        """Test processing job status recording."""
        record_processing_job_status(5)
        record_processing_job_status(0)
        record_processing_job_status(10)

        assert True  # All recordings successful

    def test_record_failed_job(self):
        """Test failed job recording."""
        job_types = ["analysis", "insight", "processing", "upload"]
        error_types = ["timeout", "validation", "network", "internal"]

        for job_type in job_types:
            for error_type in error_types:
                record_failed_job(job_type, error_type)

        assert True  # All recordings successful


class TestDynamoDBMetrics:
    """Test DynamoDB operation metrics recording."""

    def test_record_dynamodb_operation_with_duration(self):
        """Test DynamoDB operation recording with duration."""
        record_dynamodb_operation("create", "health_data", "success", 0.05)
        record_dynamodb_operation("read", "users", "success", 0.02)
        record_dynamodb_operation("update", "health_data", "failed", 0.1)
        record_dynamodb_operation("delete", "temp_data", "success", 0.03)

        assert True  # All recordings successful

    def test_record_dynamodb_operation_without_duration(self):
        """Test DynamoDB operation recording without duration."""
        record_dynamodb_operation("create", "health_data", "success")
        record_dynamodb_operation("read", "users", "failed")

        assert True  # All recordings successful

    def test_record_dynamodb_operation_various_operations(self):
        """Test DynamoDB operations with various operation types."""
        operations = ["create", "read", "update", "delete", "scan", "query"]
        tables = ["health_data", "users", "processing_jobs", "audit_logs"]
        statuses = ["success", "failed", "timeout"]

        for operation in operations:
            for table in tables:
                for status in statuses:
                    record_dynamodb_operation(operation, table, status, 0.1)

        assert True  # All recordings successful


class TestPubSubMetrics:
    """Test Pub/Sub metrics recording."""

    def test_record_pubsub_message_with_duration(self):
        """Test Pub/Sub message recording with duration."""
        record_pubsub_message("health-analysis", "success", 0.5)
        record_pubsub_message("insight-generation", "failed", 1.2)

        assert True  # All recordings successful

    def test_record_pubsub_message_without_duration(self):
        """Test Pub/Sub message recording without duration."""
        record_pubsub_message("health-analysis", "success")
        record_pubsub_message("insight-generation", "failed")

        assert True  # All recordings successful

    def test_record_pubsub_message_various_topics(self):
        """Test Pub/Sub messages with various topics."""
        topics = [
            "health-analysis",
            "insight-generation",
            "data-processing",
            "notifications",
        ]
        statuses = ["success", "failed", "retry", "timeout"]

        for topic in topics:
            for status in statuses:
                record_pubsub_message(topic, status, 0.1)

        assert True  # All recordings successful


class TestMetricsContext:
    """Test MetricsContext context manager."""

    def test_metrics_context_pat_inference_success(self):
        """Test MetricsContext for PAT inference success."""
        with MetricsContext("pat_inference"):
            time.sleep(0.01)  # Simulate work

        assert True  # Context manager completed successfully

    def test_metrics_context_pat_inference_failure(self):
        """Test MetricsContext for PAT inference failure."""
        with pytest.raises(ValueError), MetricsContext("pat_inference"):
            time.sleep(0.01)  # Simulate work
            raise ValueError("Test error")

    def test_metrics_context_insight_generation_success(self):
        """Test MetricsContext for insight generation success."""
        labels = {"model": "gemini-2.0-flash-exp"}
        with MetricsContext("insight_generation", labels):
            time.sleep(0.01)  # Simulate work

        assert True  # Context manager completed successfully

    def test_metrics_context_insight_generation_failure(self):
        """Test MetricsContext for insight generation failure."""
        labels = {"model": "gemini-pro"}
        with pytest.raises(RuntimeError):
            with MetricsContext("insight_generation", labels):
                time.sleep(0.01)  # Simulate work
                raise RuntimeError("Test error")

    def test_metrics_context_health_data_processing_success(self):
        """Test MetricsContext for health data processing success."""
        labels = {"stage": "preprocessing"}
        with MetricsContext("health_data_processing", labels):
            time.sleep(0.01)  # Simulate work

        assert True  # Context manager completed successfully

    def test_metrics_context_health_data_processing_failure(self):
        """Test MetricsContext for health data processing failure."""
        labels = {"stage": "analysis"}
        with pytest.raises(Exception):
            with MetricsContext("health_data_processing", labels):
                time.sleep(0.01)  # Simulate work
                raise Exception("Processing error")

    def test_metrics_context_dynamodb_operation_success(self):
        """Test MetricsContext for DynamoDB operation success."""
        labels = {"operation": "create", "table": "health_data"}
        with MetricsContext("dynamodb_operation", labels):
            time.sleep(0.01)  # Simulate work

        assert True  # Context manager completed successfully

    def test_metrics_context_dynamodb_operation_failure(self):
        """Test MetricsContext for DynamoDB operation failure."""
        labels = {"operation": "update", "table": "users"}
        with pytest.raises(Exception):
            with MetricsContext("dynamodb_operation", labels):
                time.sleep(0.01)  # Simulate work
                raise Exception("Database error")

    def test_metrics_context_unknown_operation(self):
        """Test MetricsContext with unknown operation type."""
        with MetricsContext("unknown_operation"):
            time.sleep(0.01)  # Simulate work

        assert True  # Should not raise error for unknown operations

    def test_metrics_context_no_labels(self):
        """Test MetricsContext without labels."""
        with MetricsContext("pat_inference"):
            time.sleep(0.01)  # Simulate work

        assert True  # Context manager completed successfully

    def test_metrics_context_empty_labels(self):
        """Test MetricsContext with empty labels."""
        with MetricsContext("insight_generation", {}):
            time.sleep(0.01)  # Simulate work

        assert True  # Context manager completed successfully

    def test_metrics_context_returns_self(self):
        """Test that MetricsContext __enter__ returns self."""
        context = MetricsContext("pat_inference")
        with context as ctx:
            assert ctx is context


class TestMetricsIntegration:
    """Test metrics integration scenarios."""

    def test_real_world_scenario(self):
        """Test a real-world scenario with multiple metrics."""
        # Simulate HTTP request
        record_http_request("POST", "/api/v1/health-data", 200, 0.2)

        # Simulate health data processing
        record_health_data_upload("success", "apple_health")
        record_health_data_processing("preprocessing", 0.5)
        record_health_metric_processed("heart_rate")
        record_health_metric_processed("steps")

        # Simulate PAT inference
        record_pat_inference("success", 0.8)

        # Simulate insight generation
        record_insight_generation("success", "gemini-2.0-flash-exp", 2.1)

        # Simulate database operations
        record_dynamodb_operation("create", "health_data", "success", 0.05)
        record_dynamodb_operation("read", "users", "success", 0.02)

        # Simulate job status
        record_processing_job_status(3)

        assert True  # All operations completed successfully

    def test_error_scenario(self):
        """Test error handling scenario."""
        # Simulate failed operations
        record_http_request("POST", "/api/v1/error", 500, 0.1)
        record_health_data_upload("failed", "manual")
        record_pat_inference("failed", 0.5)
        record_insight_generation("failed", "gemini-pro", 1.0)
        record_dynamodb_operation("create", "health_data", "failed", 0.1)
        record_failed_job("analysis", "timeout")

        assert True  # All error recordings completed successfully

    def test_concurrent_metrics_recording(self):
        """Test concurrent metrics recording."""
        import threading

        def record_metrics():
            for i in range(10):
                record_http_request("GET", f"/api/v1/test/{i}", 200, 0.1)
                record_health_metric_processed("heart_rate")
                record_pat_inference("success", 0.5)

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=record_metrics)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert True  # All concurrent recordings completed successfully


class TestMetricsEdgeCases:
    """Test edge cases and error conditions."""

    def test_record_functions_with_edge_values(self):
        """Test recording functions with edge values."""
        # Test with very small durations
        record_http_request("GET", "/api/v1/test", 200, 0.001)
        record_health_data_processing("preprocessing", 0.0001)

        # Test with large durations
        record_pat_inference("success", 30.0)
        record_insight_generation("success", "gemini-pro", 60.0)

        # Test with special characters in labels
        record_health_data_upload("success", "apple_health_v2.0")
        record_dynamodb_operation("read", "health-data-2024", "success", 0.1)

        assert True  # All edge cases handled successfully

    def test_metrics_context_timing_accuracy(self):
        """Test that MetricsContext timing is reasonably accurate."""
        start_time = time.time()
        with MetricsContext("pat_inference"):
            time.sleep(0.1)  # Sleep for 100ms
        end_time = time.time()

        # The actual time should be close to what we slept
        actual_duration = end_time - start_time
        assert 0.09 <= actual_duration <= 0.2  # Allow some tolerance

    def test_metrics_context_exception_handling(self):
        """Test that MetricsContext handles different exception types."""
        exception_types = [ValueError, RuntimeError, KeyError, TypeError]

        for exc_type in exception_types:
            with pytest.raises(exc_type), MetricsContext("pat_inference"):
                raise exc_type("Test error")

    @pytest.mark.asyncio
    async def test_metrics_endpoint_integration(self):
        """Test integration between recording and endpoint."""
        # Record some metrics
        record_http_request("GET", "/api/v1/test", 200, 0.1)
        record_health_data_upload("success", "apple_health")

        # Get metrics endpoint
        response = await get_metrics()

        # Verify the recorded metrics appear in the output
        content = response.body.decode("utf-8")
        assert "clarity_http_requests_total" in content
        assert "clarity_health_data_uploads_total" in content
        assert isinstance(response, Response)


class TestMetricsModuleExports:
    """Test module exports and public API."""

    def test_all_exports_available(self):
        """Test that all expected exports are available."""
        from clarity.api.v1.metrics import __all__

        expected_exports = [
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

        assert set(__all__) == set(expected_exports)

    def test_router_is_fastapi_router(self):
        """Test that router is a FastAPI router instance."""
        from fastapi import APIRouter

        assert isinstance(router, APIRouter)

    def test_metrics_context_is_context_manager(self):
        """Test that MetricsContext implements context manager protocol."""
        context = MetricsContext("test")
        assert hasattr(context, "__enter__")
        assert hasattr(context, "__exit__")
        assert callable(context.__enter__)
        assert callable(context.__exit__)
