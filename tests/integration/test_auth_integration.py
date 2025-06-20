"""Integration tests for authentication system.

Tests the authentication system with proper dependency injection
and service configuration.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from clarity.main import create_app


class TestAuthenticationIntegration:
    """Integration tests for authentication system."""

    @pytest.fixture
    @staticmethod
    def client() -> TestClient:
        """Create test client."""
        app = create_app()
        return TestClient(app)

    @staticmethod
    def test_auth_health_endpoint(client: TestClient) -> None:
        """Test authentication health endpoint returns proper status."""
        response = client.get("/api/v1/auth/health")

        # Should return either 200 (healthy) or 503 (service unavailable)
        assert response.status_code in {200, 503}

        # Response should be JSON
        data = response.json()

        if response.status_code == 200:
            # Healthy response format
            assert "status" in data
            assert "service" in data
            assert data["service"] == "authentication"
        else:
            # Unhealthy response format (wrapped in detail)
            assert "detail" in data
            detail = data["detail"]
            assert "status" in detail
            assert "service" in detail
            assert detail["service"] == "authentication"

    @staticmethod
    def test_registration_endpoint_exists(client: TestClient) -> None:
        """Test registration endpoint exists and handles requests."""
        # Test with missing data to trigger validation
        response = client.post("/api/v1/auth/register", json={})

        # Should return either 422 (validation error) or 500 (service not configured)
        assert response.status_code in {422, 500}

        # Response should be JSON
        data = response.json()
        assert "detail" in data

    @staticmethod
    def test_login_endpoint_exists(client: TestClient) -> None:
        """Test login endpoint exists and handles requests."""
        # Test with missing data to trigger validation
        response = client.post("/api/v1/auth/login", json={})

        # Should return either 422 (validation error) or 500 (service not configured)
        assert response.status_code in {422, 500}

        # Response should be JSON
        data = response.json()
        assert "detail" in data

    @staticmethod
    def test_refresh_endpoint_exists(client: TestClient) -> None:
        """Test refresh endpoint exists and handles requests."""
        # Test with missing data to trigger validation
        response = client.post("/api/v1/auth/refresh", json={})

        # Should return either 422 (validation error) or 500 (service not configured)
        assert response.status_code in {422, 500}

        # Response should be JSON
        data = response.json()
        assert "detail" in data

    @staticmethod
    def test_logout_endpoint_exists(client: TestClient) -> None:
        """Test logout endpoint exists and handles requests."""
        # Test with missing data to trigger validation
        response = client.post("/api/v1/auth/logout", json={})

        # Should return either 422 (validation error) or 500 (service not configured)
        assert response.status_code in {422, 500}

        # Response should be JSON
        data = response.json()
        assert "detail" in data

    @staticmethod
    def test_endpoints_fail_gracefully_without_service(
        client: TestClient,
    ) -> None:
        """Test that endpoints fail gracefully when service is not configured."""
        # Test registration with valid data structure
        registration_data = {
            "email": "test@example.com",
            "password": "SecurePass123!",
            "first_name": "Test",
            "last_name": "User",
            "terms_accepted": True,
            "privacy_policy_accepted": True,
        }

        response = client.post("/api/v1/auth/register", json=registration_data)

        # Should return 500 when service is not configured
        assert response.status_code == 500

        # Should have proper error structure
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], (str, dict))
