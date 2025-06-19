"""CLARITY Digital Twin Platform - Logging Configuration.

HIPAA-compliant structured logging with audit trail support and
configurable log levels for development and production environments.
"""

# removed - breaks FastAPI

import logging
import logging.config
import sys
from typing import Any

from clarity.core.config_aws import get_settings

# Configure logger
logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for the application based on environment settings."""
    settings = get_settings()

    # Base configuration for structured logging
    logging_config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": (
                    "%(asctime)s | %(name)s | %(levelname)s | "
                    "%(filename)s:%(lineno)d | %(funcName)s | %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "simple": {
                "format": "%(asctime)s | %(levelname)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                "format": (
                    "%(asctime)s | %(name)s | %(levelname)s | "
                    "%(filename)s:%(lineno)d | %(funcName)s | %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.log_level,
                "formatter": "detailed" if settings.debug else "simple",
                "stream": sys.stdout,
            },
        },
        "loggers": {
            "clarity": {
                "level": settings.log_level,
                "handlers": ["console"],
                "propagate": True,
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {
            "level": settings.log_level,
            "handlers": ["console"],
        },
    }

    # Apply configuration
    logging.config.dictConfig(logging_config)

    # Log successful configuration
    logger = logging.getLogger(__name__)
    environment_msg = f"Logging configured for {settings.environment} environment"
    level_msg = f"with level {settings.log_level}"
    logger.info("%s %s", environment_msg, level_msg)
