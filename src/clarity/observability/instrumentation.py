"""
OpenTelemetry Instrumentation Setup

Comprehensive auto-instrumentation with zero code changes required.
Provides distributed tracing, metrics, and context propagation.
"""
import os
import logging
from typing import Optional

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.botocore import BotocoreInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.propagators.jaeger import JaegerPropagator
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.propagators.baggage import BaggagePropagator
from opentelemetry.propagators.tracecontext import TraceContextPropagator
from prometheus_client import CollectorRegistry

logger = logging.getLogger(__name__)

# Global instances
_tracer_provider: Optional[TracerProvider] = None
_meter_provider: Optional[MeterProvider] = None
_instrumented = False


def setup_observability(
    service_name: str = "clarity-backend",
    service_version: str = "0.2.0",
    environment: str = "production",
    jaeger_endpoint: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    enable_console_export: bool = False,
) -> None:
    """
    Setup comprehensive OpenTelemetry instrumentation.
    
    Args:
        service_name: Name of the service
        service_version: Version of the service
        environment: Environment (development, staging, production)
        jaeger_endpoint: Jaeger collector endpoint
        otlp_endpoint: OTLP collector endpoint
        enable_console_export: Enable console export for debugging
    """
    global _tracer_provider, _meter_provider, _instrumented
    
    if _instrumented:
        logger.warning("Observability already setup, skipping")
        return
        
    logger.info("🔍 Setting up Clarity Observability Stack...")
    
    # Create resource
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        "environment": environment,
        "service.instance.id": os.getenv("HOSTNAME", "unknown"),
        "deployment.environment": environment,
    })
    
    # Setup tracing
    _setup_tracing(resource, jaeger_endpoint, otlp_endpoint, enable_console_export)
    
    # Setup metrics
    _setup_metrics(resource, otlp_endpoint)
    
    # Setup propagators
    _setup_propagators()
    
    # Auto-instrument libraries
    _setup_auto_instrumentation()
    
    _instrumented = True
    logger.info("✅ Clarity Observability Stack initialized successfully")


def _setup_tracing(
    resource: Resource,
    jaeger_endpoint: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    enable_console_export: bool = False,
) -> None:
    """Setup distributed tracing with multiple exporters."""
    global _tracer_provider
    
    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(_tracer_provider)
    
    span_processors = []
    
    # Jaeger exporter
    if jaeger_endpoint or os.getenv("JAEGER_ENDPOINT"):
        jaeger_exporter = JaegerExporter(
            agent_host_name=os.getenv("JAEGER_AGENT_HOST", "jaeger"),
            agent_port=int(os.getenv("JAEGER_AGENT_PORT", "6831")),
            collector_endpoint=jaeger_endpoint or os.getenv("JAEGER_ENDPOINT"),
        )
        span_processors.append(BatchSpanProcessor(jaeger_exporter))
        logger.info("✅ Jaeger trace exporter configured")
    
    # OTLP exporter
    if otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            headers=_get_otlp_headers(),
        )
        span_processors.append(BatchSpanProcessor(otlp_exporter))
        logger.info("✅ OTLP trace exporter configured")
    
    # Console exporter for debugging
    if enable_console_export or os.getenv("OTEL_TRACE_CONSOLE", "false").lower() == "true":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        console_exporter = ConsoleSpanExporter()
        span_processors.append(BatchSpanProcessor(console_exporter))
        logger.info("✅ Console trace exporter configured")
    
    # Add all processors
    for processor in span_processors:
        _tracer_provider.add_span_processor(processor)
    
    if not span_processors:
        logger.warning("⚠️  No trace exporters configured - traces will not be exported")


def _setup_metrics(
    resource: Resource,
    otlp_endpoint: Optional[str] = None,
) -> None:
    """Setup metrics collection with Prometheus and OTLP."""
    global _meter_provider
    
    readers = []
    
    # Prometheus metrics reader
    prometheus_registry = CollectorRegistry()
    prometheus_reader = PrometheusMetricReader(registry=prometheus_registry)
    readers.append(prometheus_reader)
    logger.info("✅ Prometheus metrics reader configured")
    
    # OTLP metrics exporter
    if otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        otlp_metric_exporter = OTLPMetricExporter(
            endpoint=otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            headers=_get_otlp_headers(),
        )
        otlp_reader = PeriodicExportingMetricReader(
            exporter=otlp_metric_exporter,
            export_interval_millis=10000,  # 10 seconds
        )
        readers.append(otlp_reader)
        logger.info("✅ OTLP metrics exporter configured")
    
    # Create meter provider
    _meter_provider = MeterProvider(
        resource=resource,
        metric_readers=readers,
    )
    metrics.set_meter_provider(_meter_provider)


def _setup_propagators() -> None:
    """Setup trace context propagation."""
    # Support multiple propagation formats
    propagators = [
        TraceContextPropagator(),  # W3C Trace Context
        BaggagePropagator(),       # W3C Baggage
        B3MultiFormat(),           # Zipkin B3
        JaegerPropagator(),        # Jaeger
    ]
    
    composite_propagator = CompositePropagator(propagators)
    set_global_textmap(composite_propagator)
    logger.info("✅ Trace propagators configured")


def _setup_auto_instrumentation() -> None:
    """Setup automatic instrumentation for common libraries."""
    
    # FastAPI instrumentation
    FastAPIInstrumentor().instrument()
    logger.info("✅ FastAPI auto-instrumentation enabled")
    
    # HTTP clients
    RequestsInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()
    logger.info("✅ HTTP clients auto-instrumentation enabled")
    
    # AWS services
    BotocoreInstrumentor().instrument()
    logger.info("✅ AWS Botocore auto-instrumentation enabled")
    
    # Redis (if available)
    try:
        RedisInstrumentor().instrument()
        logger.info("✅ Redis auto-instrumentation enabled")
    except Exception as e:
        logger.debug("Redis instrumentation not available: %s", e)


def _get_otlp_headers() -> dict[str, str]:
    """Get OTLP headers from environment."""
    headers = {}
    
    # API key authentication
    api_key = os.getenv("OTEL_EXPORTER_OTLP_API_KEY")
    if api_key:
        headers["api-key"] = api_key
    
    # Bearer token authentication
    token = os.getenv("OTEL_EXPORTER_OTLP_TOKEN")
    if token:
        headers["authorization"] = f"Bearer {token}"
    
    # Custom headers
    custom_headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
    if custom_headers:
        for header in custom_headers.split(","):
            if "=" in header:
                key, value = header.split("=", 1)
                headers[key.strip()] = value.strip()
    
    return headers


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer instance."""
    return trace.get_tracer(name)


def get_meter(name: str) -> metrics.Meter:
    """Get a meter instance."""
    return metrics.get_meter(name)


def is_instrumented() -> bool:
    """Check if observability is already setup."""
    return _instrumented