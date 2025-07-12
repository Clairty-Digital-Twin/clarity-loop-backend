"""Test the test endpoints."""

from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from clarity.api.v1.test import router


@pytest.fixture
def test_app():
    """Create test app with test router."""
    app = FastAPI()
    app.include_router(router, prefix="/test")
    return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


def test_simple_ping_no_auth(client):
    """Test simple ping endpoint without auth."""
    response = client.get("/test/ping")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "PONG! Backend is alive!"
    assert data["path"] == "/test/ping"
    assert data["has_auth_header"] is False
    assert data["auth_header_preview"] == "NO AUTH HEADER"


def test_simple_ping_with_auth(client):
    """Test simple ping endpoint with auth header."""
    response = client.get(
        "/test/ping", headers={"Authorization": "Bearer test-token-123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "PONG! Backend is alive!"
    assert data["has_auth_header"] is True
    assert data["auth_header_preview"] == "Bearer test-token-123"


def test_check_middleware_no_user(client):
    """Test check middleware endpoint without user."""
    response = client.get("/test/check-middleware")
    assert response.status_code == 200
    data = response.json()
    assert data["middleware_ran"] is False
    assert data["user_info"] is None
    assert data["auth_header"] == "NO AUTH HEADER"


def test_check_middleware_with_user(test_app, client):
    """Test check middleware endpoint with user in state."""

    # Create a middleware that sets user
    @test_app.middleware("http")
    async def add_user_middleware(request, call_next):
        # Add a mock user to request state
        mock_user = Mock()
        mock_user.user_id = "test-user-123"
        request.state.user = mock_user
        return await call_next(request)

    response = client.get(
        "/test/check-middleware", headers={"Authorization": "Bearer token"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["middleware_ran"] is True
    assert data["user_info"]["exists"] is True
    assert data["user_info"]["user_id"] == "test-user-123"
    assert "Bearer token" in data["auth_header"]
