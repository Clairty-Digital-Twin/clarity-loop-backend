"""
Enhanced Metrics Collection

Comprehensive metrics for API performance, system health, and ML models.
Uses both Prometheus and OpenTelemetry metrics for maximum compatibility.
"""
import time
import psutil
import logging
from typing import Dict, Optional, List, Any
from contextlib import contextmanager
from functools import wraps

from prometheus_client import (
    Counter, Histogram, Gauge, Info, Enum,
    CollectorRegistry, generate_latest
)
from opentelemetry import metrics
from opentelemetry.metrics import Observation

logger = logging.getLogger(__name__)


class ClarityMetrics:
    """Comprehensive metrics collection for Clarity backend."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.meter = metrics.get_meter(__name__)
        
        # Initialize all metrics
        self._init_api_metrics()
        self._init_system_metrics()
        self._init_ml_metrics()
        self._init_business_metrics()
        self._init_security_metrics()
        
        logger.info("✅ Clarity metrics initialized")
    
    def _init_api_metrics(self) -> None:
        """Initialize API performance metrics."""
        # Request metrics
        self.requests_total = Counter(
            "clarity_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            "clarity_http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
            registry=self.registry
        )
        
        self.request_size = Histogram(
            "clarity_http_request_size_bytes",
            "HTTP request size in bytes",
            ["method", "endpoint"],
            registry=self.registry
        )
        
        self.response_size = Histogram(
            "clarity_http_response_size_bytes", 
            "HTTP response size in bytes",
            ["method", "endpoint"],
            registry=self.registry
        )
        
        # Active connections
        self.active_connections = Gauge(
            "clarity_active_connections",
            "Number of active connections",
            registry=self.registry
        )
        
        # WebSocket metrics
        self.websocket_connections = Gauge(
            "clarity_websocket_connections_active",
            "Active WebSocket connections",
            registry=self.registry
        )
        
        self.websocket_messages = Counter(
            "clarity_websocket_messages_total",
            "Total WebSocket messages",
            ["direction", "message_type"],
            registry=self.registry
        )
    
    def _init_system_metrics(self) -> None:
        """Initialize system health metrics."""
        # Process metrics
        self.process_cpu_usage = Gauge(
            "clarity_process_cpu_usage_percent",
            "Process CPU usage percentage",
            registry=self.registry
        )
        
        self.process_memory_usage = Gauge(
            "clarity_process_memory_usage_bytes",
            "Process memory usage in bytes",
            registry=self.registry
        )
        
        self.process_open_fds = Gauge(
            "clarity_process_open_file_descriptors",
            "Number of open file descriptors",
            registry=self.registry
        )
        
        # System metrics
        self.system_cpu_usage = Gauge(
            "clarity_system_cpu_usage_percent",
            "System CPU usage percentage",
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            "clarity_system_memory_usage_bytes",
            "System memory usage in bytes",
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            "clarity_system_disk_usage_bytes",
            "System disk usage in bytes",
            ["device"],
            registry=self.registry
        )
        
        # Python runtime metrics
        self.python_gc_collections = Counter(
            "clarity_python_gc_collections_total",
            "Python garbage collector collections",
            ["generation"],
            registry=self.registry
        )
        
        self.python_gc_time = Histogram(
            "clarity_python_gc_time_seconds",
            "Time spent in garbage collection",
            registry=self.registry
        )
        
        # Application info
        self.app_info = Info(
            "clarity_app_info",
            "Application information",
            registry=self.registry
        )
        
        self.app_health = Enum(
            "clarity_app_health_status",
            "Application health status",
            states=["healthy", "degraded", "unhealthy"],
            registry=self.registry
        )
    
    def _init_ml_metrics(self) -> None:
        """Initialize ML model metrics."""
        # Model inference metrics
        self.model_predictions_total = Counter(
            "clarity_ml_predictions_total",
            "Total ML model predictions",
            ["model_name", "model_version", "status"],
            registry=self.registry
        )
        
        self.model_inference_duration = Histogram(
            "clarity_ml_inference_duration_seconds",
            "ML model inference duration",
            ["model_name", "model_version"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
            registry=self.registry
        )
        
        self.model_queue_size = Gauge(
            "clarity_ml_queue_size",
            "ML processing queue size",
            ["model_name"],
            registry=self.registry  
        )
        
        self.model_cache_hit_rate = Gauge(
            "clarity_ml_cache_hit_rate",
            "ML model cache hit rate",
            ["model_name"],
            registry=self.registry
        )
        
        # PAT model specific metrics
        self.pat_analysis_duration = Histogram(
            "clarity_pat_analysis_duration_seconds",
            "PAT analysis processing time",
            ["analysis_type", "data_size_bucket"],
            registry=self.registry
        )
        
        self.pat_model_accuracy = Gauge(
            "clarity_pat_model_accuracy",
            "PAT model accuracy score",
            ["model_size", "data_type"],
            registry=self.registry
        )
    
    def _init_business_metrics(self) -> None:
        """Initialize business-specific metrics."""
        # User activity
        self.active_users = Gauge(
            "clarity_active_users",
            "Number of active users",
            ["time_window"],
            registry=self.registry
        )
        
        self.user_sessions = Counter(
            "clarity_user_sessions_total", 
            "Total user sessions",
            ["session_type"],
            registry=self.registry
        )
        
        # Health data processing
        self.health_data_points = Counter(
            "clarity_health_data_points_total",
            "Total health data points processed",
            ["data_type", "source"],
            registry=self.registry
        )
        
        self.health_data_processing_duration = Histogram(
            "clarity_health_data_processing_duration_seconds",
            "Health data processing duration",
            ["data_type", "processing_stage"],
            registry=self.registry
        )
        
        # Insights generation
        self.insights_generated = Counter(
            "clarity_insights_generated_total",
            "Total insights generated",
            ["insight_type", "confidence_level"],
            registry=self.registry
        )
        
        self.insight_accuracy = Gauge(
            "clarity_insight_accuracy_score",
            "Insight accuracy score",
            ["insight_type"],
            registry=self.registry
        )
    
    def _init_security_metrics(self) -> None:
        """Initialize security-related metrics."""
        # Authentication metrics
        self.auth_attempts = Counter(
            "clarity_auth_attempts_total",
            "Authentication attempts",
            ["status", "method"],
            registry=self.registry
        )
        
        self.auth_failures = Counter(
            "clarity_auth_failures_total",
            "Authentication failures",
            ["reason", "source_ip"],
            registry=self.registry
        )
        
        self.account_lockouts = Counter(
            "clarity_account_lockouts_total",
            "Account lockouts",
            ["reason"],
            registry=self.registry
        )
        
        # Rate limiting
        self.rate_limit_hits = Counter(
            "clarity_rate_limit_hits_total",
            "Rate limit hits",
            ["endpoint", "client_type"],
            registry=self.registry
        )
        
        self.suspicious_activity = Counter(
            "clarity_suspicious_activity_total",
            "Suspicious activity detected",
            ["activity_type", "severity"],
            registry=self.registry
        )
    
    # Convenience methods for common operations
    def record_request(self, method: str, endpoint: str, status_code: int, 
                      duration: float, request_size: int = 0, response_size: int = 0) -> None:
        """Record HTTP request metrics."""
        self.requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=str(status_code)
        ).inc()
        
        self.request_duration.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
        
        if request_size > 0:
            self.request_size.labels(
                method=method, 
                endpoint=endpoint
            ).observe(request_size)
        
        if response_size > 0:
            self.response_size.labels(
                method=method, 
                endpoint=endpoint
            ).observe(response_size)
    
    def record_ml_prediction(self, model_name: str, model_version: str, 
                           duration: float, status: str = "success") -> None:
        """Record ML prediction metrics."""
        self.model_predictions_total.labels(
            model_name=model_name,
            model_version=model_version,
            status=status
        ).inc()
        
        self.model_inference_duration.labels(
            model_name=model_name,
            model_version=model_version
        ).observe(duration)
    
    @contextmanager
    def time_ml_inference(self, model_name: str, model_version: str):
        """Context manager to time ML inference."""
        start_time = time.time()
        try:
            yield
            duration = time.time() - start_time
            self.record_ml_prediction(model_name, model_version, duration, "success")
        except Exception as e:
            duration = time.time() - start_time
            self.record_ml_prediction(model_name, model_version, duration, "error")
            raise
    
    def update_system_metrics(self) -> None:
        """Update system health metrics."""
        try:
            # Process metrics
            process = psutil.Process()
            self.process_cpu_usage.set(process.cpu_percent())
            self.process_memory_usage.set(process.memory_info().rss)
            
            try:
                self.process_open_fds.set(process.num_fds())
            except (psutil.AttributeError, psutil.AccessDenied):
                # Not available on all platforms
                pass
            
            # System metrics
            self.system_cpu_usage.set(psutil.cpu_percent())
            self.system_memory_usage.set(psutil.virtual_memory().used)
            
            # Disk usage
            for disk in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(disk.mountpoint)
                    self.system_disk_usage.labels(device=disk.device).set(usage.used)
                except (psutil.PermissionError, OSError):
                    continue
                    
        except Exception as e:
            logger.warning("Failed to update system metrics: %s", e)
    
    def set_app_info(self, version: str, environment: str, **kwargs) -> None:
        """Set application information."""
        info_dict = {
            "version": version,
            "environment": environment,
            **kwargs
        }
        self.app_info.info(info_dict)
    
    def set_health_status(self, status: str) -> None:
        """Set application health status."""
        if status in ["healthy", "degraded", "unhealthy"]:
            self.app_health.state(status)
        else:
            logger.warning("Invalid health status: %s", status)
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.registry).decode('utf-8')


# Global metrics instance
_metrics_instance: Optional[ClarityMetrics] = None


def get_metrics() -> ClarityMetrics:
    """Get the global metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = ClarityMetrics()
    return _metrics_instance


def metrics_middleware():
    """Decorator to add metrics to functions."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                # Record success metrics
                return result
            except Exception as e:
                duration = time.time() - start_time
                # Record error metrics
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                # Record success metrics
                return result
            except Exception as e:
                duration = time.time() - start_time
                # Record error metrics
                raise
        
        return async_wrapper if hasattr(func, '__code__') and func.__code__.co_flags & 0x80 else sync_wrapper
    return decorator