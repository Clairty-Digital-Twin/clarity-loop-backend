"""Feature flag monitoring and metrics dashboard."""

import time
from typing import Any

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from clarity.core.feature_flags_integration import get_feature_flag_health

# Create router for metrics endpoints
router = APIRouter(prefix="/metrics/feature-flags", tags=["monitoring"])


class FeatureFlagMetrics(BaseModel):
    """Feature flag metrics model."""

    refresh_success_total: int
    refresh_failure_total: dict[str, int]
    stale_config: bool
    last_refresh_timestamp: float | None
    circuit_breaker_state: str
    avg_refresh_duration_seconds: float
    p95_refresh_duration_seconds: float
    p99_refresh_duration_seconds: float


class FeatureFlagAlert(BaseModel):
    """Feature flag alert model."""

    severity: str  # critical, warning, info
    message: str
    timestamp: float
    details: dict[str, Any]


@router.get("/prometheus")
async def prometheus_metrics() -> Response:
    """Export feature flag metrics in Prometheus format."""
    # Generate metrics in Prometheus format
    metrics_output = generate_latest()

    return Response(
        content=metrics_output,
        media_type=CONTENT_TYPE_LATEST,
    )


@router.get("/dashboard", response_model=FeatureFlagMetrics)
async def metrics_dashboard() -> FeatureFlagMetrics:
    """Get feature flag metrics dashboard data."""
    # Get current health
    health = get_feature_flag_health()

    # Calculate metrics
    # Note: In production, these would be fetched from actual Prometheus metrics
    return FeatureFlagMetrics(
        refresh_success_total=100,  # Placeholder
        refresh_failure_total={
            "network_error": 5,
            "timeout": 2,
            "circuit_breaker_open": 1,
        },
        stale_config=health["config_stale"],
        last_refresh_timestamp=time.time() - (health["config_age_seconds"] or 0),
        circuit_breaker_state=health["circuit_breaker_state"],
        avg_refresh_duration_seconds=0.5,  # Placeholder
        p95_refresh_duration_seconds=0.8,  # Placeholder
        p99_refresh_duration_seconds=1.2,  # Placeholder
    )


@router.get("/alerts", response_model=list[FeatureFlagAlert])
async def get_alerts() -> list[FeatureFlagAlert]:
    """Get active feature flag alerts."""
    alerts = []
    health = get_feature_flag_health()

    # Check for stale configuration
    if health["config_stale"]:
        config_age = health["config_age_seconds"] or 0
        alerts.append(
            FeatureFlagAlert(
                severity="warning" if config_age < 600 else "critical",
                message=f"Feature flag configuration is stale ({config_age:.0f}s old)",
                timestamp=time.time(),
                details={
                    "config_age_seconds": config_age,
                    "last_refresh": health["last_refresh"],
                },
            )
        )

    # Check circuit breaker state
    if health["circuit_breaker_state"] == "open":
        alerts.append(
            FeatureFlagAlert(
                severity="critical",
                message="Feature flag config store circuit breaker is OPEN",
                timestamp=time.time(),
                details={
                    "circuit_breaker_state": "open",
                    "refresh_failures": health["refresh_failures"],
                },
            )
        )
    elif health["circuit_breaker_state"] == "half-open":
        alerts.append(
            FeatureFlagAlert(
                severity="warning",
                message="Feature flag config store circuit breaker is HALF-OPEN",
                timestamp=time.time(),
                details={
                    "circuit_breaker_state": "half-open",
                    "refresh_failures": health["refresh_failures"],
                },
            )
        )

    # Check refresh failures
    if health["refresh_failures"] > 5:
        alerts.append(
            FeatureFlagAlert(
                severity="warning",
                message=f"High number of refresh failures: {health['refresh_failures']}",
                timestamp=time.time(),
                details={
                    "refresh_failures": health["refresh_failures"],
                },
            )
        )

    return alerts


@router.get("/grafana")
async def grafana_dashboard_config() -> dict[str, Any]:
    """Get Grafana dashboard configuration for feature flags."""
    return {
        "dashboard": {
            "title": "Feature Flags Monitoring",
            "panels": [
                {
                    "title": "Refresh Success Rate",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "rate(feature_flag_refresh_success_total[5m])",
                            "legendFormat": "Success Rate",
                        },
                        {
                            "expr": "rate(feature_flag_refresh_failure_total[5m])",
                            "legendFormat": "Failure Rate",
                        },
                    ],
                },
                {
                    "title": "Configuration Staleness",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "feature_flag_stale_config",
                            "legendFormat": "Stale Config",
                        },
                    ],
                    "thresholds": [
                        {"value": 0, "color": "green"},
                        {"value": 1, "color": "red"},
                    ],
                },
                {
                    "title": "Circuit Breaker State",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "feature_flag_circuit_breaker_state",
                            "legendFormat": "State",
                        },
                    ],
                    "mappings": [
                        {"value": 0, "text": "Closed", "color": "green"},
                        {"value": 1, "text": "Open", "color": "red"},
                        {"value": 2, "text": "Half-Open", "color": "yellow"},
                    ],
                },
                {
                    "title": "Refresh Duration",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.5, feature_flag_refresh_duration_seconds_bucket)",
                            "legendFormat": "p50",
                        },
                        {
                            "expr": "histogram_quantile(0.95, feature_flag_refresh_duration_seconds_bucket)",
                            "legendFormat": "p95",
                        },
                        {
                            "expr": "histogram_quantile(0.99, feature_flag_refresh_duration_seconds_bucket)",
                            "legendFormat": "p99",
                        },
                    ],
                },
                {
                    "title": "Time Since Last Refresh",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "time() - feature_flag_last_refresh_timestamp",
                            "legendFormat": "Seconds",
                        },
                    ],
                    "unit": "s",
                    "thresholds": [
                        {"value": 60, "color": "green"},
                        {"value": 300, "color": "yellow"},
                        {"value": 600, "color": "red"},
                    ],
                },
            ],
            "refresh": "10s",
            "time": {
                "from": "now-1h",
                "to": "now",
            },
        },
    }


@router.post("/simulate-failure")
async def simulate_failure() -> dict[str, str]:
    """Simulate a config store failure for testing monitoring."""
    # This would be used in testing/staging environments
    # to verify monitoring and alerting work correctly
    return {
        "message": "This endpoint would simulate failures in non-production environments",
        "warning": "Not implemented for safety - use test environments only",
    }
