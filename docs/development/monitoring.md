# Monitoring & Observability

This document outlines monitoring, logging, and observability for the Clarity Loop Backend to ensure system health, performance, and reliability.

## Observability Stack

### Three Pillars of Observability

```
┌─────────────────┬─────────────────┬─────────────────┐
│     METRICS     │      LOGS       │     TRACES      │
├─────────────────┼─────────────────┼─────────────────┤
│ • System health │ • Event logs    │ • Request flow  │
│ • Performance   │ • Error logs    │ • Latency       │
│ • Business KPIs │ • Audit logs    │ • Dependencies  │
│ • SLA metrics   │ • Access logs   │ • Bottlenecks   │
└─────────────────┴─────────────────┴─────────────────┘
```

### Technology Stack

- **Metrics**: Google Cloud Monitoring (formerly Stackdriver)
- **Logging**: Google Cloud Logging with structured logs
- **Tracing**: Google Cloud Trace with OpenTelemetry
- **Alerting**: Google Cloud Alerting + PagerDuty
- **Dashboards**: Google Cloud Console + Custom dashboards

## Application Monitoring

### FastAPI Metrics Integration

```python
# src/clarity/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import FastAPI, Request, Response
import time
import structlog

# Metrics definitions
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_USERS = Gauge(
    'active_users_total',
    'Number of active users'
)

HEALTH_DATA_POINTS = Counter(
    'health_data_points_total',
    'Total health data points processed',
    ['data_type', 'source']
)

ML_INFERENCE_TIME = Histogram(
    'ml_inference_duration_seconds',
    'ML model inference time',
    ['model_name', 'model_version']
)

class MetricsMiddleware:
    def __init__(self, app: FastAPI):
        self.app = app
        self.logger = structlog.get_logger("metrics")
    
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        endpoint = request.url.path
        method = request.method
        status_code = response.status_code
        
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        # Log slow requests
        if duration > 2.0:
            await self.logger.awarning(
                "Slow request detected",
                endpoint=endpoint,
                method=method,
                duration=duration,
                status_code=status_code
            )
        
        return response

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        generate_latest(),
        media_type="text/plain"
    )
```

### Business Metrics Tracking

```python
# src/clarity/monitoring/business_metrics.py
from google.cloud import monitoring_v3
from datetime import datetime
import asyncio

class BusinessMetrics:
    def __init__(self):
        self.client = monitoring_v3.MetricServiceClient()
        self.project_name = f"projects/clarity-loop-backend"
    
    async def track_user_engagement(self, user_id: str, action: str):
        """Track user engagement metrics."""
        series = monitoring_v3.TimeSeries()
        series.metric.type = "custom.googleapis.com/user/engagement"
        series.metric.labels["action"] = action
        series.metric.labels["user_id"] = user_id
        
        point = monitoring_v3.Point()
        point.value.int64_value = 1
        point.interval.end_time.seconds = int(time.time())
        series.points = [point]
        
        await asyncio.to_thread(
            self.client.create_time_series,
            name=self.project_name,
            time_series=[series]
        )
    
    async def track_health_data_quality(self, quality_score: float, data_type: str):
        """Track health data quality metrics."""
        series = monitoring_v3.TimeSeries()
        series.metric.type = "custom.googleapis.com/health_data/quality"
        series.metric.labels["data_type"] = data_type
        
        point = monitoring_v3.Point()
        point.value.double_value = quality_score
        point.interval.end_time.seconds = int(time.time())
        series.points = [point]
        
        await asyncio.to_thread(
            self.client.create_time_series,
            name=self.project_name,
            time_series=[series]
        )
    
    async def track_ai_insights_generation(
        self, 
        processing_time: float,
        model_name: str,
        success: bool
    ):
        """Track AI insights generation metrics."""
        # Processing time metric
        series = monitoring_v3.TimeSeries()
        series.metric.type = "custom.googleapis.com/ai/processing_time"
        series.metric.labels["model"] = model_name
        series.metric.labels["success"] = str(success)
        
        point = monitoring_v3.Point()
        point.value.double_value = processing_time
        point.interval.end_time.seconds = int(time.time())
        series.points = [point]
        
        await asyncio.to_thread(
            self.client.create_time_series,
            name=self.project_name,
            time_series=[series]
        )
```

## Structured Logging

### Logging Configuration

```python
# src/clarity/monitoring/logging.py
import structlog
import logging
from google.cloud import logging as cloud_logging
from pythonjsonlogger import jsonlogger

# Configure structured logging
def configure_logging():
    """Configure structured logging for production."""
    
    # Google Cloud Logging client
    cloud_client = cloud_logging.Client()
    cloud_client.setup_logging()
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# Custom logging middleware
class LoggingMiddleware:
    def __init__(self, app: FastAPI):
        self.app = app
        self.logger = structlog.get_logger("api")
    
    async def __call__(self, request: Request, call_next):
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        
        # Add to request state
        request.state.correlation_id = correlation_id
        
        # Log request
        await self.logger.ainfo(
            "Request started",
            correlation_id=correlation_id,
            method=request.method,
            url=str(request.url),
            user_agent=request.headers.get("user-agent"),
            ip_address=request.client.host
        )
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log response
            await self.logger.ainfo(
                "Request completed",
                correlation_id=correlation_id,
                status_code=response.status_code,
                duration=duration
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            await self.logger.aerror(
                "Request failed",
                correlation_id=correlation_id,
                error=str(e),
                duration=duration,
                exc_info=True
            )
            raise
```

### Health Data Processing Logs

```python
# src/clarity/monitoring/health_data_logging.py
import structlog
from typing import Dict, Any

class HealthDataLogger:
    def __init__(self):
        self.logger = structlog.get_logger("health_data")
    
    async def log_data_ingestion(
        self,
        user_id: str,
        data_type: str,
        record_count: int,
        source: str,
        quality_score: float
    ):
        """Log health data ingestion events."""
        await self.logger.ainfo(
            "Health data ingested",
            user_id=user_id,
            data_type=data_type,
            record_count=record_count,
            source=source,
            quality_score=quality_score,
            event_type="data_ingestion"
        )
    
    async def log_ml_processing(
        self,
        user_id: str,
        model_name: str,
        input_size: int,
        processing_time: float,
        success: bool,
        error: str = None
    ):
        """Log ML model processing events."""
        log_data = {
            "event_type": "ml_processing",
            "user_id": user_id,
            "model_name": model_name,
            "input_size": input_size,
            "processing_time": processing_time,
            "success": success
        }
        
        if error:
            log_data["error"] = error
            await self.logger.aerror("ML processing failed", **log_data)
        else:
            await self.logger.ainfo("ML processing completed", **log_data)
    
    async def log_insights_generation(
        self,
        user_id: str,
        insight_type: str,
        model_confidence: float,
        generation_time: float
    ):
        """Log AI insights generation events."""
        await self.logger.ainfo(
            "AI insights generated",
            user_id=user_id,
            insight_type=insight_type,
            model_confidence=model_confidence,
            generation_time=generation_time,
            event_type="insights_generation"
        )
```

## Distributed Tracing

### OpenTelemetry Integration

```python
# src/clarity/monitoring/tracing.py
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

def configure_tracing(app: FastAPI):
    """Configure distributed tracing."""
    
    # Set up tracer provider
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Configure Cloud Trace exporter
    cloud_trace_exporter = CloudTraceSpanExporter()
    span_processor = BatchSpanProcessor(cloud_trace_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)
    
    # Instrument HTTP clients
    RequestsInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()
    
    return tracer

# Custom tracing decorators
def trace_function(span_name: str = None):
    """Decorator to trace function execution."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            name = span_name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_as_current_span(name) as span:
                try:
                    # Add function metadata
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    
                    result = await func(*args, **kwargs)
                    span.set_attribute("function.success", True)
                    return result
                    
                except Exception as e:
                    span.set_attribute("function.success", False)
                    span.set_attribute("function.error", str(e))
                    span.record_exception(e)
                    raise
                    
        return wrapper
    return decorator

# Usage example
class HealthDataProcessor:
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
    
    @trace_function("health_data.process_batch")
    async def process_batch(self, user_id: str, data_batch: List[Dict]):
        """Process batch of health data with tracing."""
        
        with self.tracer.start_as_current_span("validate_data") as span:
            span.set_attribute("batch.size", len(data_batch))
            span.set_attribute("user.id", user_id)
            
            # Validation logic
            validated_data = await self._validate_batch(data_batch)
        
        with self.tracer.start_as_current_span("store_data") as span:
            span.set_attribute("validated.count", len(validated_data))
            
            # Storage logic
            await self._store_batch(user_id, validated_data)
        
        return len(validated_data)
```

## Health Checks & Uptime Monitoring

### Comprehensive Health Checks

```python
# src/clarity/monitoring/health.py
from fastapi import APIRouter, HTTPException
from google.cloud import firestore, storage
import asyncio
import httpx

router = APIRouter()

class HealthChecker:
    def __init__(self):
        self.firestore_client = firestore.AsyncClient()
        self.storage_client = storage.Client()
    
    async def check_database(self) -> Dict[str, Any]:
        """Check Firestore connectivity and performance."""
        try:
            start_time = time.time()
            
            # Simple read operation
            doc_ref = self.firestore_client.collection("health_check").document("test")
            await doc_ref.get()
            
            duration = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time": duration,
                "service": "firestore"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "service": "firestore"
            }
    
    async def check_storage(self) -> Dict[str, Any]:
        """Check Cloud Storage connectivity."""
        try:
            start_time = time.time()
            
            # List buckets operation
            buckets = list(self.storage_client.list_buckets())
            
            duration = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time": duration,
                "service": "cloud_storage",
                "bucket_count": len(buckets)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "service": "cloud_storage"
            }
    
    async def check_ml_models(self) -> Dict[str, Any]:
        """Check ML model availability."""
        try:
            # Mock health check for ML models
            # In production, this would call actual model endpoints
            start_time = time.time()
            
            # Simulate model health check
            await asyncio.sleep(0.1)
            
            duration = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time": duration,
                "service": "ml_models",
                "models": ["actigraphy_transformer", "gemini_insights"]
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "service": "ml_models"
            }

@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "clarity-loop-backend"
    }

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with dependency status."""
    checker = HealthChecker()
    
    # Run all health checks concurrently
    checks = await asyncio.gather(
        checker.check_database(),
        checker.check_storage(),
        checker.check_ml_models(),
        return_exceptions=True
    )
    
    # Process results
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": "healthy",
        "services": {}
    }
    
    for check in checks:
        if isinstance(check, Exception):
            results["overall_status"] = "degraded"
            results["services"]["unknown"] = {
                "status": "error",
                "error": str(check)
            }
        else:
            service_name = check["service"]
            results["services"][service_name] = check
            
            if check["status"] != "healthy":
                results["overall_status"] = "degraded"
    
    # Return appropriate status code
    if results["overall_status"] != "healthy":
        raise HTTPException(status_code=503, detail=results)
    
    return results

@router.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe endpoint."""
    # Check if application is ready to serve requests
    return {"status": "ready"}

@router.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe endpoint."""
    # Check if application is alive
    return {"status": "alive"}
```

## Alerting & Incident Response

### Alert Configuration

```python
# src/clarity/monitoring/alerts.py
from google.cloud import monitoring_v3
from typing import List, Dict

class AlertManager:
    def __init__(self):
        self.client = monitoring_v3.AlertPolicyServiceClient()
        self.project_name = f"projects/clarity-loop-backend"
    
    def create_sla_alerts(self) -> List[str]:
        """Create SLA-based alert policies."""
        policies = []
        
        # API Response Time Alert
        response_time_policy = monitoring_v3.AlertPolicy(
            display_name="API Response Time SLA",
            combiner=monitoring_v3.AlertPolicy.ConditionCombiner.OR,
            conditions=[
                monitoring_v3.AlertPolicy.Condition(
                    display_name="API response time > 2s",
                    condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                        filter='metric.type="custom.googleapis.com/http/request_duration"',
                        comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                        threshold_value=monitoring_v3.TypedValue(double_value=2.0),
                        duration={"seconds": 300},  # 5 minutes
                        aggregations=[
                            monitoring_v3.Aggregation(
                                alignment_period={"seconds": 60},
                                per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN,
                                cross_series_reducer=monitoring_v3.Aggregation.Reducer.REDUCE_MEAN
                            )
                        ]
                    )
                )
            ],
            notification_channels=[
                "projects/clarity-loop-backend/notificationChannels/CHANNEL_ID"
            ]
        )
        
        policies.append(self.client.create_alert_policy(
            name=self.project_name,
            alert_policy=response_time_policy
        ))
        
        # Error Rate Alert
        error_rate_policy = monitoring_v3.AlertPolicy(
            display_name="High Error Rate",
            combiner=monitoring_v3.AlertPolicy.ConditionCombiner.OR,
            conditions=[
                monitoring_v3.AlertPolicy.Condition(
                    display_name="Error rate > 5%",
                    condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                        filter='metric.type="custom.googleapis.com/http/error_rate"',
                        comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                        threshold_value=monitoring_v3.TypedValue(double_value=0.05),
                        duration={"seconds": 300}
                    )
                )
            ]
        )
        
        policies.append(self.client.create_alert_policy(
            name=self.project_name,
            alert_policy=error_rate_policy
        ))
        
        return [policy.name for policy in policies]

# Incident response automation
class IncidentResponder:
    def __init__(self):
        self.logger = structlog.get_logger("incident_response")
    
    async def handle_high_error_rate(self, alert_data: Dict):
        """Handle high error rate incidents."""
        await self.logger.acritical(
            "High error rate detected",
            error_rate=alert_data.get("error_rate"),
            affected_endpoints=alert_data.get("endpoints"),
            incident_id=alert_data.get("incident_id")
        )
        
        # Auto-scaling logic
        await self._trigger_auto_scaling()
        
        # Circuit breaker activation
        await self._activate_circuit_breakers()
    
    async def handle_service_down(self, service_name: str):
        """Handle service outage incidents."""
        await self.logger.acritical(
            "Service outage detected",
            service=service_name,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Failover logic
        await self._initiate_failover(service_name)
```

## Performance Monitoring

### SLA Metrics & Dashboards

```yaml
# monitoring/sla_objectives.yaml
service_level_objectives:
  api_availability:
    target: 99.9%
    measurement_window: "30d"
    
  api_latency:
    target: "95% < 500ms"
    measurement_window: "24h"
    
  data_processing_latency:
    target: "90% < 30s"
    measurement_window: "1h"
    
  ml_inference_latency:
    target: "95% < 2s"
    measurement_window: "1h"

error_budgets:
  monthly_downtime: "43.2m"  # 99.9% uptime
  daily_error_budget: "144s"  # 0.1% of day
```

### Dashboard Configuration

```python
# monitoring/dashboards.py
DASHBOARD_CONFIG = {
    "overview": {
        "widgets": [
            "api_requests_per_second",
            "api_response_time_p95",
            "error_rate",
            "active_users",
            "system_health"
        ]
    },
    "health_data": {
        "widgets": [
            "data_ingestion_rate",
            "data_quality_score",
            "processing_latency",
            "storage_utilization"
        ]
    },
    "ml_performance": {
        "widgets": [
            "model_inference_time",
            "model_accuracy",
            "prediction_confidence",
            "model_resource_usage"
        ]
    },
    "security": {
        "widgets": [
            "authentication_failures",
            "rate_limit_violations",
            "suspicious_activity",
            "access_patterns"
        ]
    }
}
```

This comprehensive monitoring strategy ensures the Clarity Loop Backend maintains high availability, performance, and reliability while providing deep insights into system behavior and user experience.
