"""Global pytest configuration for logging setup.

This file ensures consistent logging behavior across all tests
and prevents caplog issues caused by logger configuration conflicts.
"""

import logging

import pytest


@pytest.fixture(autouse=True)
def setup_logging() -> None:
    """Global fixture to ensure consistent logging configuration across all tests.

    This fixture:
    - Sets up proper logging levels for our modules
    - Ensures caplog can capture logs consistently
    - Prevents logger configuration conflicts between tests
    """
    # Ensure our main loggers are properly configured
    loggers_to_configure = [
        "clarity.core.decorators",
        "clarity.services.dynamodb_service",
        "clarity.core.cloud",
        "clarity.api",
        "clarity.ml",
        "clarity.entrypoints",
    ]

    for logger_name in loggers_to_configure:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        # Ensure propagation is enabled so caplog can capture messages
        logger.propagate = True

    # Configure root logger to ensure caplog works
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # No cleanup needed - pytest handles caplog cleanup


@pytest.fixture(autouse=True)
def configure_caplog(caplog: pytest.LogCaptureFixture) -> None:
    """Global fixture to configure caplog for all tests.

    This ensures caplog can capture logs from all our modules
    without needing manual setup in each test.
    """
    # Set caplog to capture DEBUG level logs from all our modules
    caplog.set_level(logging.DEBUG)

    # Specifically configure for our main modules
    caplog.set_level(logging.DEBUG, logger="clarity.core.decorators")
    caplog.set_level(logging.DEBUG, logger="clarity.services.dynamodb_service")
    caplog.set_level(logging.DEBUG, logger="clarity.core.cloud")
    caplog.set_level(logging.DEBUG, logger="clarity.api")
    caplog.set_level(logging.DEBUG, logger="clarity.ml")
    caplog.set_level(logging.DEBUG, logger="clarity.entrypoints")
