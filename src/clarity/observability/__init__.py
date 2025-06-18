"""
Clarity Observability Stack

A comprehensive production-grade observability solution with:
- OpenTelemetry distributed tracing
- Structured logging with correlation IDs
- Enhanced Prometheus metrics
- Alert management
- Custom dashboards

This module provides zero-code-change instrumentation for complete observability.
"""

from .instrumentation import setup_observability, get_tracer, get_meter
from .correlation import CorrelationMiddleware, get_correlation_id
from .metrics import ClarityMetrics
from .logging import setup_structured_logging

__all__ = [
    "setup_observability",
    "get_tracer", 
    "get_meter",
    "CorrelationMiddleware",
    "get_correlation_id",
    "ClarityMetrics",
    "setup_structured_logging"
]