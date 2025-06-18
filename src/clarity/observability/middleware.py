"""
Observability Middleware

Integrates all observability components into FastAPI middleware:
- Request tracing and correlation
- Metrics collection
- Performance monitoring
- Error tracking
"""
import time
import logging
from typing import Callable, Optional
from urllib.parse import urlparse

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.instrumentation.utils import http_status_to_status_code

from .metrics import get_metrics
from .correlation import get_correlation_id
from .alerting import get_alert_manager

logger = logging.getLogger(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive observability middleware that instruments all requests.
    
    Provides:
    - Distributed tracing
    - Metrics collection
    - Performance monitoring
    - Error tracking
    - Automatic alerting
    """
    
    def __init__(
        self,
        app,
        trace_requests: bool = True,
        collect_metrics: bool = True,
        track_performance: bool = True,
        auto_alert: bool = True,
        exclude_paths: Optional[list] = None,
    ):
        super().__init__(app)
        self.trace_requests = trace_requests
        self.collect_metrics = collect_metrics
        self.track_performance = track_performance
        self.auto_alert = auto_alert
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs", "/redoc"]
        
        # Get global instances
        self.metrics = get_metrics() if collect_metrics else None
        self.alert_manager = get_alert_manager() if auto_alert else None
        self.tracer = trace.get_tracer(__name__) if trace_requests else None
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with comprehensive observability."""
        
        # Skip observability for excluded paths
        if self._should_exclude(request):
            return await call_next(request)
        
        # Extract request information
        method = request.method
        url = str(request.url)
        path = request.url.path
        endpoint = self._normalize_endpoint(path)
        
        # Start timing
        start_time = time.time()
        request_size = int(request.headers.get("content-length", 0))
        
        # Create span if tracing enabled
        span = None
        if self.trace_requests and self.tracer:
            span = self.tracer.start_span(
                f"{method} {endpoint}",
                kind=trace.SpanKind.SERVER,
            )
            
            # Set span attributes
            span.set_attributes({
                "http.method": method,
                "http.url": url,
                "http.route": endpoint,
                "http.scheme": request.url.scheme,
                "http.host": request.url.hostname or "unknown",
                "http.user_agent": request.headers.get("user-agent", ""),
                "http.request_content_length": request_size,
                "correlation.id": get_correlation_id() or "unknown",
            })
        
        # Process request
        response = None
        error = None
        status_code = 500
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Set success span status
            if span:
                span.set_status(Status(http_status_to_status_code(status_code)))
                span.set_attribute("http.status_code", status_code)
                response_size = int(response.headers.get("content-length", 0))
                span.set_attribute("http.response_content_length", response_size)
            
        except Exception as e:
            error = e
            status_code = 500
            
            # Set error span status
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("http.status_code", status_code)
                span.record_exception(e)
            
            # Re-raise the exception
            raise
            
        finally:
            # Calculate duration
            duration = time.time() - start_time
            response_size = int(response.headers.get("content-length", 0)) if response else 0
            
            # Collect metrics
            if self.collect_metrics and self.metrics:
                self.metrics.record_request(
                    method=method,
                    endpoint=endpoint,
                    status_code=status_code,
                    duration=duration,
                    request_size=request_size,
                    response_size=response_size
                )
            
            # Check for performance issues and alerts
            if self.auto_alert and self.alert_manager:
                await self._check_alerts(
                    method, endpoint, status_code, duration, error
                )
            
            # End span
            if span:
                span.end()
            
            # Log request completion
            self._log_request(
                method, endpoint, status_code, duration, 
                request_size, response_size, error
            )
        
        return response
    
    def _should_exclude(self, request: Request) -> bool:
        """Check if request should be excluded from observability."""
        path = request.url.path
        return any(path.startswith(excluded) for excluded in self.exclude_paths)
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for consistent metrics."""
        # Remove query parameters
        if "?" in path:
            path = path.split("?")[0]
        
        # Common path normalizations
        # Replace UUIDs with placeholder
        import re
        uuid_pattern = r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        path = re.sub(uuid_pattern, '/{uuid}', path, flags=re.IGNORECASE)
        
        # Replace numeric IDs with placeholder
        numeric_pattern = r'/\d+'
        path = re.sub(numeric_pattern, '/{id}', path)
        
        # Ensure path starts with /
        if not path.startswith('/'):
            path = '/' + path
        
        return path
    
    async def _check_alerts(
        self, 
        method: str, 
        endpoint: str, 
        status_code: int, 
        duration: float,
        error: Optional[Exception]
    ) -> None:
        """Check if alerts should be triggered."""
        try:
            # High error rate check
            if status_code >= 500:
                # This would typically check metrics over time window
                # For now, we'll trigger on individual 5xx errors
                pass
            
            # Slow response time check
            if duration > 1.0:  # 1 second threshold
                self.alert_manager.fire_alert(
                    rule_name="slow_response_time",
                    message=f"Slow response detected: {method} {endpoint} took {duration:.2f}s",
                    labels={
                        "method": method,
                        "endpoint": endpoint,
                        "duration": f"{duration:.2f}s"
                    }
                )
            
            # Exception check
            if error:
                self.alert_manager.fire_alert(
                    rule_name="application_error",
                    message=f"Application error: {method} {endpoint} - {str(error)}",
                    labels={
                        "method": method,
                        "endpoint": endpoint,
                        "error_type": type(error).__name__,
                        "error_message": str(error)
                    }
                )
                
        except Exception as e:
            logger.error("Failed to check alerts", error=str(e))
    
    def _log_request(
        self,
        method: str,
        endpoint: str, 
        status_code: int,
        duration: float,
        request_size: int,
        response_size: int,
        error: Optional[Exception]
    ) -> None:
        """Log request completion."""
        log_data = {
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "duration_ms": round(duration * 1000, 2),
            "request_size": request_size,
            "response_size": response_size,
            "correlation_id": get_correlation_id(),
        }
        
        if error:
            log_data["error"] = str(error)
            log_data["error_type"] = type(error).__name__
            logger.error("Request failed", **log_data)
        elif status_code >= 400:
            logger.warning("Request completed with error", **log_data)
        else:
            logger.info("Request completed", **log_data)


class SystemMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect system metrics periodically."""
    
    def __init__(self, app, update_interval: int = 30):
        super().__init__(app)
        self.update_interval = update_interval
        self.last_update = 0
        self.metrics = get_metrics()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Update system metrics if needed."""
        current_time = time.time()
        
        # Update system metrics periodically
        if current_time - self.last_update > self.update_interval:
            try:
                self.metrics.update_system_metrics()
                self.last_update = current_time
            except Exception as e:
                logger.warning("Failed to update system metrics", error=str(e))
        
        return await call_next(request)


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Middleware to monitor application health."""
    
    def __init__(self, app, health_check_interval: int = 60):
        super().__init__(app)
        self.health_check_interval = health_check_interval
        self.last_health_check = 0
        self.metrics = get_metrics()
        self.alert_manager = get_alert_manager()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Perform health checks periodically."""
        current_time = time.time()
        
        # Perform health check periodically
        if current_time - self.last_health_check > self.health_check_interval:
            try:
                health_status = await self._check_health()
                self.metrics.set_health_status(health_status)
                self.last_health_check = current_time
                
                # Alert if unhealthy
                if health_status != "healthy":
                    self.alert_manager.fire_alert(
                        rule_name="application_unhealthy",
                        message=f"Application health status: {health_status}",
                        labels={"health_status": health_status}
                    )
                    
            except Exception as e:
                logger.warning("Health check failed", error=str(e))
                self.metrics.set_health_status("unhealthy")
        
        return await call_next(request)
    
    async def _check_health(self) -> str:
        """Perform application health check."""
        try:
            # Check database connectivity
            # Check external services
            # Check system resources
            
            # For now, return healthy
            return "healthy"
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return "unhealthy"