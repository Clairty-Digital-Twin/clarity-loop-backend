"""
Structured Logging Setup

Production-grade structured logging with:
- JSON formatting for log aggregation
- Sensitive data masking
- Correlation ID integration
- Dynamic log level control
- Performance optimization
"""
import os
import sys
import json
import logging
import logging.config
from typing import Dict, Any, Optional, List
from datetime import datetime
import re

import structlog
from structlog.types import FilteringBoundLogger
from rich.logging import RichHandler
from rich.console import Console

from .correlation import get_correlation_id


# Sensitive data patterns to mask
SENSITIVE_PATTERNS = [
    # Passwords and tokens
    (re.compile(r'"password":\s*"[^"]*"', re.IGNORECASE), '"password": "***"'),
    (re.compile(r'"token":\s*"[^"]*"', re.IGNORECASE), '"token": "***"'),
    (re.compile(r'"api_key":\s*"[^"]*"', re.IGNORECASE), '"api_key": "***"'),
    (re.compile(r'"secret":\s*"[^"]*"', re.IGNORECASE), '"secret": "***"'),
    (re.compile(r'"access_token":\s*"[^"]*"', re.IGNORECASE), '"access_token": "***"'),
    (re.compile(r'"refresh_token":\s*"[^"]*"', re.IGNORECASE), '"refresh_token": "***"'),
    
    # Credit cards and SSNs
    (re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'), '****-****-****-****'),
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '***-**-****'),
    
    # Email addresses (partial masking)
    (re.compile(r'\b([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'), 
     lambda m: f"{m.group(1)[:2]}***@{m.group(2)}"),
    
    # Phone numbers
    (re.compile(r'\b\d{3}-\d{3}-\d{4}\b'), '***-***-****'),
    (re.compile(r'\b\(\d{3}\)\s*\d{3}-\d{4}\b'), '(***) ***-****'),
]


class SensitiveDataMasker:
    """Mask sensitive data in log messages."""
    
    def __init__(self, patterns: List[tuple] = None):
        self.patterns = patterns or SENSITIVE_PATTERNS
    
    def mask(self, message: str) -> str:
        """Mask sensitive data in the message."""
        if not isinstance(message, str):
            return message
            
        for pattern, replacement in self.patterns:
            if callable(replacement):
                message = pattern.sub(replacement, message)
            else:
                message = pattern.sub(replacement, message)
        return message


class ClarityJSONFormatter(logging.Formatter):
    """Custom JSON formatter with correlation ID and sensitive data masking."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masker = SensitiveDataMasker()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Extract correlation ID
        correlation_id = get_correlation_id()
        
        # Base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": self.masker.mask(record.getMessage()),
            "correlation_id": correlation_id,
            "thread_id": record.thread,
            "thread_name": record.threadName,
            "process_id": record.process,
            "module": record.module,
            "function": record.funcName,
            "line_number": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                'filename', 'module', 'lineno', 'funcName', 'created', 
                'msecs', 'relativeCreated', 'thread', 'threadName', 'processName',
                'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info'
            }:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        # Add service metadata
        log_entry["service"] = {
            "name": "clarity-backend",
            "version": "0.2.0",
            "environment": os.getenv("ENVIRONMENT", "production")
        }
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class ClarityStructlogProcessor:
    """Structlog processor for Clarity-specific fields."""
    
    def __init__(self):
        self.masker = SensitiveDataMasker()
    
    def __call__(self, logger: FilteringBoundLogger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process log event."""
        # Add correlation ID
        correlation_id = get_correlation_id()
        if correlation_id:
            event_dict["correlation_id"] = correlation_id
        
        # Add service metadata
        event_dict["service"] = {
            "name": "clarity-backend",
            "version": "0.2.0",
            "environment": os.getenv("ENVIRONMENT", "production")
        }
        
        # Mask sensitive data in the event
        if "event" in event_dict:
            event_dict["event"] = self.masker.mask(str(event_dict["event"]))
        
        # Mask sensitive data in other string fields
        for key, value in event_dict.items():
            if isinstance(value, str):
                event_dict[key] = self.masker.mask(value)
        
        return event_dict


def setup_structured_logging(
    level: str = "INFO",
    json_logs: bool = True,
    enable_rich: bool = False,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Setup structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Use JSON formatting for logs
        enable_rich: Enable rich console output for development
        log_file: Path to log file (optional)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    # Determine log level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Setup structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        ClarityStructlogProcessor(),
    ]
    
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Setup standard logging
    handlers = []
    
    # Console handler
    if enable_rich and not json_logs:
        console = Console(
            stderr=True,
            force_terminal=True,
            width=120,
        )
        console_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_path=False,
            show_time=True,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        if json_logs:
            console_handler.setFormatter(ClarityJSONFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)8s] [%(correlation_id)s] %(name)s: %(message)s"
                )
            )
    
    console_handler.setLevel(numeric_level)
    handlers.append(console_handler)
    
    # File handler (if specified)
    if log_file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(ClarityJSONFormatter())
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        format="%(message)s"
    )
    
    # Configure specific loggers
    _configure_loggers(numeric_level)
    
    # Add correlation filter
    from .correlation import CorrelationFilter
    correlation_filter = CorrelationFilter()
    for handler in handlers:
        handler.addFilter(correlation_filter)
    
    # Log setup completion
    logger = structlog.get_logger(__name__)
    logger.info(
        "Structured logging configured",
        level=level,
        json_logs=json_logs,
        enable_rich=enable_rich,
        log_file=log_file,
    )


def _configure_loggers(level: int) -> None:
    """Configure specific loggers."""
    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("s3transfer").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("opentelemetry").setLevel(logging.INFO)
    
    # Clarity-specific loggers
    logging.getLogger("clarity").setLevel(level)


def get_logger(name: str) -> FilteringBoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def set_log_level(level: str) -> None:
    """Dynamically change log level."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger().setLevel(numeric_level)
    
    # Update all handlers
    for handler in logging.getLogger().handlers:
        handler.setLevel(numeric_level)
    
    logger = get_logger(__name__)
    logger.info("Log level changed", new_level=level)


def add_sensitive_pattern(pattern: str, replacement: str) -> None:
    """Add a new sensitive data pattern."""
    compiled_pattern = re.compile(pattern, re.IGNORECASE)
    SENSITIVE_PATTERNS.append((compiled_pattern, replacement))
    
    logger = get_logger(__name__)
    logger.info("Added sensitive data pattern", pattern=pattern)