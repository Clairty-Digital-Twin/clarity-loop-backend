"""🧪 Pytest Configuration and Shared Fixtures

Provides shared test fixtures, configuration, and utilities for all test modules.
"""

import asyncio
from collections.abc import AsyncGenerator
from collections.abc import Generator
import os
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient
import pytest

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
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_document = MagicMock()
    
    mock_client.collection.return_value = mock_collection
    mock_collection.document.return_value = mock_document
    mock_document.get.return_value.exists = True
    mock_document.get.return_value.to_dict.return_value = {"test": "data"}
    
    return mock_client


@pytest.fixture
def mock_firebase_auth():
    """Mock Firebase Auth for testing."""
    mock_auth = MagicMock()
    mock_auth.verify_id_token.return_value = {
        "uid": "test-user-id",
        "email": "test@example.com",
        "email_verified": True
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
    import torch
    
    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.return_value = torch.tensor([[0.1, 0.2, 0.7]])  # Mock predictions
    
    # Mock attention weights for explainability
    mock_attention = torch.rand(1, 8, 560, 560)  # [batch, heads, seq, seq]
    mock_model.get_attention_weights.return_value = mock_attention
    
    return mock_model


@pytest.fixture
def sample_actigraphy_data():
    """Sample actigraphy data for testing."""
    import numpy as np
    
    # 7 days of minute-level data (10,080 minutes)
    return {
        "user_id": "test-user-123",
        "start_date": "2024-01-01T00:00:00Z",
        "end_date": "2024-01-07T23:59:00Z",
        "activity_counts": np.random.poisson(10, 10080).tolist(),
        "timestamps": [
            f"2024-01-{day:02d}T{hour:02d}:{minute:02d}:00Z"
            for day in range(1, 8)
            for hour in range(24)
            for minute in range(60)
        ]
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
        "sleep_efficiency": 0.85
    }


@pytest.fixture
async def app() -> FastAPI:
    """Create FastAPI application for testing."""
    from clarity.main import create_app
    
    app = create_app()
    return app


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
        "Content-Type": "application/json"
    }


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    from unittest.mock import MagicMock

    import redis
    
    mock_client = MagicMock(spec=redis.Redis)
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.delete.return_value = 1
    mock_client.exists.return_value = False
    
    return mock_client


@pytest.fixture(autouse=True)
def mock_environment_variables(monkeypatch):
    """Mock environment variables for testing."""
    test_env = {
        "TESTING": "1",
        "ENVIRONMENT": "testing",
        "DEBUG": "true",
        "SECRET_KEY": "test-secret-key",
        "JWT_SECRET_KEY": "test-jwt-secret",
        "FIRESTORE_EMULATOR_HOST": "localhost:8080",
        "FIREBASE_AUTH_EMULATOR_HOST": "localhost:9099",
        "REDIS_URL": "redis://localhost:6379/1",
        "GCP_PROJECT_ID": "test-project",
        "GEMINI_API_KEY": "test-gemini-key",
        "PAT_MODEL_PATH": "./tests/fixtures/mock-pat-model"
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)


# Pytest markers for test organization
pytest_plugins = ["pytest_asyncio"]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "ml: Machine learning tests")
    config.addinivalue_line("markers", "pat: PAT model tests")
    config.addinivalue_line("markers", "gemini: Gemini AI tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "auth: Authentication tests")
    config.addinivalue_line("markers", "api: API endpoint tests")
    config.addinivalue_line("markers", "data: Data processing tests")
