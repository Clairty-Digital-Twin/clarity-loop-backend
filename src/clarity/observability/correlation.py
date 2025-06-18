"""
Correlation ID Middleware

Provides request correlation tracking across distributed systems.
Automatically generates unique correlation IDs and propagates them
through logs, traces, and response headers.
"""
import uuid
import logging
from contextvars import ContextVar
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from opentelemetry import trace, baggage

logger = logging.getLogger(__name__)

# Context variable for correlation ID
_correlation_id_context: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)

# Standard header names for correlation ID
CORRELATION_ID_HEADER = "X-Correlation-ID"
REQUEST_ID_HEADER = "X-Request-ID"
TRACE_ID_HEADER = "X-Trace-ID"


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Middleware to handle correlation IDs across requests."""
    
    def __init__(
        self,
        app,
        header_name: str = CORRELATION_ID_HEADER,
        generate_if_missing: bool = True,
        include_in_response: bool = True,
        include_trace_id: bool = True,
    ):
        super().__init__(app)
        self.header_name = header_name
        self.generate_if_missing = generate_if_missing
        self.include_in_response = include_in_response
        self.include_trace_id = include_trace_id
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with correlation ID tracking."""
        
        # Extract correlation ID from headers
        correlation_id = self._extract_correlation_id(request)
        
        # Generate new ID if not present
        if not correlation_id and self.generate_if_missing:
            correlation_id = self._generate_correlation_id()
        
        # Set correlation ID in context
        if correlation_id:
            token = _correlation_id_context.set(correlation_id)
            
            # Add to OpenTelemetry baggage
            baggage.set_baggage("correlation.id", correlation_id)
            
            # Add to current span
            span = trace.get_current_span()
            if span.is_recording():
                span.set_attribute("correlation.id", correlation_id)
                span.set_attribute("http.request.correlation_id", correlation_id)
            
            try:
                # Process request
                response = await call_next(request)
                
                # Add correlation ID to response headers
                if self.include_in_response:
                    response.headers[self.header_name] = correlation_id
                    response.headers[REQUEST_ID_HEADER] = correlation_id
                
                # Add trace ID if enabled
                if self.include_trace_id and span:
                    trace_id = format(span.get_span_context().trace_id, "032x")
                    response.headers[TRACE_ID_HEADER] = trace_id
                
                return response
                
            finally:
                # Reset context
                _correlation_id_context.reset(token)
        else:
            # No correlation ID - process normally
            return await call_next(request)
    
    def _extract_correlation_id(self, request: Request) -> Optional[str]:
        """Extract correlation ID from request headers."""
        # Check multiple possible header names
        header_names = [
            self.header_name,
            CORRELATION_ID_HEADER,
            REQUEST_ID_HEADER,
            "X-Request-Id",
            "X-Correlation-Id",
            "Correlation-ID",
            "Request-ID",
        ]
        
        for header_name in header_names:
            correlation_id = request.headers.get(header_name)
            if correlation_id:
                return correlation_id.strip()
        
        return None
    
    def _generate_correlation_id(self) -> str:
        """Generate a new correlation ID."""
        return str(uuid.uuid4())


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from context."""
    return _correlation_id_context.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID in the current context."""
    _correlation_id_context.set(correlation_id)
    
    # Also set in OpenTelemetry baggage
    baggage.set_baggage("correlation.id", correlation_id)
    
    # Add to current span
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("correlation.id", correlation_id)


class CorrelationFilter(logging.Filter):
    """Logging filter to add correlation ID to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to log record."""
        correlation_id = get_correlation_id()
        record.correlation_id = correlation_id or "unknown"
        return True


def setup_correlation_logging() -> None:
    """Setup correlation ID in logging."""
    # Add filter to root logger
    root_logger = logging.getLogger()
    correlation_filter = CorrelationFilter()
    root_logger.addFilter(correlation_filter)
    
    # Update log format to include correlation ID
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)8s] [%(correlation_id)s] %(name)s: %(message)s"
    )
    
    # Apply to all handlers
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)