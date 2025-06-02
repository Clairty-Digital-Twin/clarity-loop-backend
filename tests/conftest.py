"""Shared test fixtures and configuration for CLARITY test suite.

Provides common test utilities following pytest best practices.
"""

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from httpx import AsyncClient
import numpy as np
import pytest
from pytest import Config, MonkeyPatch
import redis
import torch

from clarity.main import create_app

# Set testing environment
os.environ["TESTING"] = "1"
os.environ["ENVIRONMENT"] = "testing"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_firestore():
    """Mock Firestore client for testing."""
    mock_db = MagicMock()

    # Mock collection and document structure
    mock_collection = MagicMock()
    mock_document = MagicMock()
    mock_document.set.return_value = None
    mock_document.get.return_value.exists = True
    mock_document.get.return_value.to_dict.return_value = {}

    mock_collection.document.return_value = mock_document
    mock_db.collection.return_value = mock_collection

    return mock_db


@pytest.fixture
def mock_firebase_auth():
    """Mock Firebase Auth for testing."""
    mock_auth = MagicMock()
    mock_auth.verify_id_token.return_value = {
        "uid": "test-user-123",
        "email": "test@example.com",
        "email_verified": True,
        "custom_claims": {"role": "patient"},
    }
    return mock_auth


@pytest.fixture
def mock_gcs_client():
    """Mock Google Cloud Storage client."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()

    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    mock_blob.upload_from_string.return_value = None

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
    activity_counts = []
    for h in range(24):
        for m in range(60):
            if h >= 22 or h <= 6:  # Night hours
                # Low activity during sleep
                activity = rng.poisson(5)
            else:  # Day hours
                # Higher activity during wake hours
                activity = rng.poisson(50)
            activity_counts.append(activity)

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


@pytest.fixture(autouse=True)
def mock_environment_variables(monkeypatch: MonkeyPatch):
    """Mock environment variables for testing."""
    test_env = {
        "TESTING": "1",
        "ENVIRONMENT": "test",
        "DEBUG": "true",
        "DATABASE_URL": "sqlite:///test.db",
        "FIREBASE_PROJECT_ID": "test-project",
        "FIREBASE_CREDENTIALS": "test-credentials.json",
        "JWT_SECRET_KEY": "test-secret-key-for-testing-only",
        "LOG_LEVEL": "DEBUG",
        "CORS_ORIGINS": "http://localhost:3000,http://localhost:8000",
        "SKIP_EXTERNAL_SERVICES": "true",  # Skip Firebase/Firestore in tests
    }

    for key, value in test_env.items():
        monkeypatch.setenv(key, value)


# Pytest markers for test organization
pytest_plugins = ["pytest_asyncio"]


def pytest_configure(config: Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "auth: Authentication related tests")
    config.addinivalue_line("markers", "database: Database related tests")


@pytest.fixture
def sample_health_metrics() -> list[dict]:
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
def sample_user_context() -> dict:
    """Provide sample user context for testing."""
    return {
        "user_id": "test-user-123",
        "email": "test@example.com",
        "roles": ["patient"],
        "permissions": ["read_own_data", "write_own_data"],
        "verified": True,
    }


@pytest.fixture
def sample_biometric_data() -> dict:
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
