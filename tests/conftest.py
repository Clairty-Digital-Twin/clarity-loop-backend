"""Shared test fixtures and configuration for CLARITY test suite.

Provides common test utilities following pytest best practices.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Remove all Google Cloud imports - we're using AWS now
from httpx import AsyncClient
import numpy as np
import pytest
import redis
import torch

from clarity.api.v1.websocket.connection_manager import ConnectionManager
from clarity.main import create_app

# Load test environment variables from .env.test
# This file should be created locally by developers and not version controlled.
# It's loaded early to ensure variables are available for other module-level setups if needed.
load_dotenv(".env.test")

# Set testing environment
os.environ["TESTING"] = "1"
os.environ["ENVIRONMENT"] = "testing"
# Skip external services in tests
os.environ["SKIP_EXTERNAL_SERVICES"] = "true"
os.environ["SKIP_AWS_INIT"] = "true"


# AWS test environment setup
os.environ["AWS_REGION"] = "us-east-1"  # Changed from AWS_DEFAULT_REGION
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"  # Keep both for compatibility
os.environ["AWS_ACCESS_KEY_ID"] = "test-key-id"
os.environ["AWS_SECRET_ACCESS_KEY"] = (
    "test-secret-key"  # noqa: S105 - Test AWS credentials
)
os.environ["COGNITO_USER_POOL_ID"] = "us-east-1_testpool123"  # Valid format
os.environ["COGNITO_CLIENT_ID"] = "test-client-id-with-enough-length"  # 20+ chars
os.environ["DYNAMODB_TABLE_PREFIX"] = "test"
# Add SECRET_KEY for config validation
os.environ["SECRET_KEY"] = "test-secret-key-for-testing"  # noqa: S105


@pytest.fixture(scope="session")
def test_env_credentials() -> dict[str, str]:
    """Provides test credentials loaded from the .env.test file or defaults."""
    return {
        "default_username": os.getenv("TEST_DEFAULT_USERNAME", "test_user@example.com"),
        "default_password": os.getenv("TEST_DEFAULT_PASSWORD", "SecurePassword123!"),
        "mock_access_token": os.getenv(
            "TEST_MOCK_ACCESS_TOKEN", "mock-access-token-value"
        ),
        "mock_refresh_token": os.getenv(
            "TEST_MOCK_REFRESH_TOKEN", "mock-refresh-token-value"
        ),
        "mock_new_access_token": os.getenv(
            "TEST_MOCK_NEW_ACCESS_TOKEN", "mock-new-access-token-value"
        ),
        "mock_new_refresh_token": os.getenv(
            "TEST_MOCK_NEW_REFRESH_TOKEN", "mock-new-refresh-token-value"
        ),
        "mock_cognito_token": os.getenv(
            "TEST_MOCK_COGNITO_TOKEN", "mock-cognito-token-value"
        ),
        "mock_sync_token": os.getenv("TEST_MOCK_SYNC_TOKEN", "mock-sync-token-value"),
    }


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_dynamodb():
    """Mock DynamoDB client for testing."""
    mock_db = MagicMock()
    mock_table = MagicMock()

    # Mock basic DynamoDB operations
    mock_table.put_item.return_value = {"ResponseMetadata": {"HTTPStatusCode": 200}}
    mock_table.get_item.return_value = {
        "Item": {"id": {"S": "test-id"}, "data": {"S": "test-data"}}
    }
    mock_table.scan.return_value = {"Items": [], "Count": 0}
    mock_table.query.return_value = {"Items": [], "Count": 0}

    mock_db.Table.return_value = mock_table
    return mock_db


@pytest.fixture
def mock_cognito_auth():
    """Mock AWS Cognito Auth for testing."""
    mock_auth = MagicMock()
    mock_auth.admin_get_user.return_value = {
        "UserAttributes": [
            {"Name": "sub", "Value": "test-user-123"},
            {"Name": "email", "Value": "test@example.com"},
            {"Name": "email_verified", "Value": "true"},
            {"Name": "custom:role", "Value": "patient"},
        ],
        "Username": "test-user-123",
    }
    return mock_auth


@pytest.fixture
def mock_s3_client():
    """Mock S3 client for testing."""
    mock_client = MagicMock()

    # Mock S3 operations
    mock_client.put_object.return_value = {"ETag": "test-etag"}
    mock_client.get_object.return_value = {"Body": MagicMock()}
    mock_client.list_objects_v2.return_value = {"Contents": []}

    return mock_client


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini AI client for testing."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.text = "This is a test AI response with health insights."
    mock_client.generate_content.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_pat_model():
    """Mock PAT (Pretrained Actigraphy Transformer) model."""
    mock_model = MagicMock()
    mock_model.predict.return_value = {
        "sleep_stages": ["awake", "light", "deep", "rem"],
        "confidence": 0.85,
        "features": torch.randn(10),
    }
    return mock_model


@pytest.fixture
def sample_actigraphy_data() -> dict[str, Any]:
    """Provide sample actigraphy data for testing."""
    # Create a random number generator for reproducible tests
    rng = np.random.RandomState(42)

    # Generate 24 hours of synthetic actigraphy data (1 sample per minute)
    timestamps = [
        f"2024-01-01T{h:02d}:{m:02d}:00Z" for h in range(24) for m in range(60)
    ]

    # Simulate sleep pattern: low activity during night hours
    activity_counts: list[int] = []  # Explicit type annotation
    for h in range(24):
        for _ in range(60):  # Use underscore for unused variable
            # Use ternary operator as suggested by SIM108
            activity = rng.poisson(5) if h >= 22 or h <= 6 else rng.poisson(50)
            activity_counts.append(int(activity))

    return {
        "user_id": "test-user-123",
        "device_id": "actigraph-001",
        "timestamps": timestamps[:100],  # First 100 samples for testing
        "activity_counts": activity_counts[:100],
        "sampling_rate": "1_min",
        "metadata": {
            "device_model": "ActiGraph GT3X+",
            "firmware_version": "1.2.3",
            "recording_duration": "24h",
        },
    }


@pytest.fixture
def sample_health_labels():
    """Sample health labels for testing."""
    return {
        "depression": 0,  # No depression (PHQ-9 < 10)
        "sleep_abnormality": 1,  # Sleep abnormality detected
        "sleep_disorder": 0,  # No sleep disorder
        "phq9_score": 5,
        "avg_sleep_hours": 6.2,
        "sleep_efficiency": 0.85,
    }


@pytest.fixture
def app() -> FastAPI:
    """Create FastAPI test application."""
    return create_app()


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client for synchronous testing."""
    return TestClient(app)


@pytest.fixture
async def async_client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client for asynchronous testing."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_headers():
    """Mock authentication headers."""
    return {
        "Authorization": "Bearer test-jwt-token",
        "Content-Type": "application/json",
    }


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock_client = MagicMock(spec=redis.Redis)
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.delete.return_value = 1
    mock_client.exists.return_value = False
    return mock_client


@pytest.fixture
def mock_test_connection_manager():
    """Stateful mock connection manager for WebSocket testing."""
    return ConnectionManager(
        heartbeat_interval=5,
        max_connections_per_user=2,
        connection_timeout=30,
        message_rate_limit=10,
        max_message_size=1024,
    )


@pytest.fixture(autouse=True)
def mock_environment_variables(monkeypatch: pytest.MonkeyPatch):
    """Mock environment variables for testing."""
    test_env = {
        "TESTING": "1",
        "ENVIRONMENT": "testing",
        "DEBUG": "true",
        "DATABASE_URL": "sqlite:///test.db",
        "AWS_REGION": "us-east-1",
        "COGNITO_USER_POOL_ID": "us-east-1_testpool123",
        "COGNITO_CLIENT_ID": "test-client-id-with-enough-length",
        "DYNAMODB_TABLE_PREFIX": "test",
        "JWT_SECRET_KEY": "test-secret-key-for-testing-only",  # This is a test secret
        "LOG_LEVEL": "DEBUG",
        "CORS_ORIGINS": '["http://localhost:3000", "http://localhost:8000"]',  # Fixed: JSON format
        "SKIP_EXTERNAL_SERVICES": "true",  # Skip external AWS services in tests
    }

    for key, value in test_env.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def sample_health_metrics() -> list[dict[str, Any]]:
    """Provide sample health metrics for testing."""
    return [
        {
            "metric_type": "heart_rate",
            "value": 72.0,
            "unit": "bpm",
            "timestamp": "2024-01-01T12:00:00Z",
            "metadata": {"device": "fitness_tracker"},
        },
        {
            "metric_type": "steps",
            "value": 8500.0,
            "unit": "count",
            "timestamp": "2024-01-01T12:00:00Z",
            "metadata": {"device": "smartphone"},
        },
        {
            "metric_type": "sleep_duration",
            "value": 7.5,
            "unit": "hours",
            "timestamp": "2024-01-01T08:00:00Z",
            "metadata": {"sleep_quality": "good"},
        },
    ]


@pytest.fixture
def sample_user_context() -> dict[str, Any]:
    """Provide sample user context for testing."""
    return {
        "user_id": "test-user-123",
        "email": "test@example.com",
        "roles": ["patient"],
        "permissions": ["read_own_data", "write_own_data"],
        "verified": True,
    }


@pytest.fixture
def sample_biometric_data() -> dict[str, Any]:
    """Provide sample biometric data for testing."""
    return {
        "user_id": "test-user-123",
        "measurements": [
            {
                "type": "blood_pressure",
                "systolic": 120,
                "diastolic": 80,
                "timestamp": "2024-01-01T09:00:00Z",
                "device": "omron_bp_monitor",
            },
            {
                "type": "weight",
                "value": 70.5,
                "unit": "kg",
                "timestamp": "2024-01-01T08:00:00Z",
                "device": "smart_scale",
            },
        ],
    }


# Pytest markers for test organization
pytest_plugins = ["pytest_asyncio"]


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "auth: Authentication related tests")
    config.addinivalue_line("markers", "database: Database related tests")
