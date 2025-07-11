"""Comprehensive test suite for debug API endpoints.

Tests both /debug/capture-raw-request and /debug/echo-login endpoints
with comprehensive coverage of JSON parsing, error handling, logging,
and edge cases to achieve 95%+ coverage.
"""

import json
import logging
from unittest.mock import patch, MagicMock
from typing import Any

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from clarity.api.v1.debug import router, capture_raw_request, echo_login_request


class TestDebugAPIModule:
    """Test the debug API module structure and configuration."""

    def test_router_configuration(self):
        """Test that the debug router is configured correctly."""
        assert router.prefix == "/debug"
        assert len(router.routes) == 2

    def test_endpoints_exist(self):
        """Test that both debug endpoints exist."""
        route_paths = [route.path for route in router.routes]
        assert "/debug/capture-raw-request" in route_paths
        assert "/debug/echo-login" in route_paths

    def test_all_endpoints_are_post(self):
        """Test that all debug endpoints use POST method."""
        for route in router.routes:
            assert "POST" in route.methods


class TestCaptureRawRequestEndpoint:
    """Test the /debug/capture-raw-request endpoint."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock request object."""
        request = MagicMock(spec=Request)
        request.method = "POST"
        request.url = "http://test.com/debug/capture-raw-request"
        request.headers = {}
        return request

    @pytest.mark.asyncio
    async def test_capture_valid_json_request(self, mock_request):
        """Test capturing a valid JSON request."""
        # Arrange
        json_data = {"username": "testuser", "password": "testpass"}
        json_bytes = json.dumps(json_data).encode("utf-8")
        
        # Create async mock for body() method
        async def mock_body():
            return json_bytes
        
        mock_request.body = mock_body
        mock_request.headers = {"Content-Type": "application/json"}

        # Act
        response = await capture_raw_request(mock_request)

        # Assert
        assert isinstance(response, dict)
        assert response["method"] == "POST"
        assert response["url"] == "http://test.com/debug/capture-raw-request"
        assert response["headers"]["Content-Type"] == "application/json"
        assert response["body_info"]["length_bytes"] == len(json_bytes)
        assert response["body_info"]["decoded_success"] is True
        assert response["body_info"]["json_valid"] is True
        assert response["body_info"]["json_data"] == json_data

    @pytest.mark.asyncio
    async def test_capture_invalid_json_request(self, mock_request):
        """Test capturing a request with invalid JSON."""
        # Arrange
        invalid_json = '{"username": "testuser", "password": "testpass"'  # Missing closing brace
        json_bytes = invalid_json.encode("utf-8")
        
        async def mock_body():
            return json_bytes
        
        mock_request.body = mock_body

        # Act
        response = await capture_raw_request(mock_request)

        # Assert
        assert response["body_info"]["decoded_success"] is True
        assert response["body_info"]["json_valid"] is False
        assert "json_error" in response["body_info"]
        assert "message" in response["body_info"]["json_error"]
        assert "position" in response["body_info"]["json_error"]

    @pytest.mark.asyncio
    async def test_capture_invalid_utf8_request(self, mock_request):
        """Test capturing a request with invalid UTF-8 encoding."""
        # Arrange
        invalid_utf8 = b'\x80\x81\x82\x83'  # Invalid UTF-8 sequence
        
        async def mock_body():
            return invalid_utf8
        
        mock_request.body = mock_body

        # Act
        response = await capture_raw_request(mock_request)

        # Assert
        assert response["body_info"]["decoded_success"] is False
        assert "decode_error" in response["body_info"]
        assert "raw_bytes" in response["body_info"]

    @pytest.mark.asyncio
    async def test_capture_empty_request(self, mock_request):
        """Test capturing an empty request."""
        # Arrange
        async def mock_body():
            return b""
        
        mock_request.body = mock_body

        # Act
        response = await capture_raw_request(mock_request)

        # Assert
        assert response["body_info"]["length_bytes"] == 0
        assert response["body_info"]["decoded_success"] is True
        assert response["body_info"]["json_valid"] is False
        assert response["body_info"]["decoded_string"] == ""

    @pytest.mark.asyncio
    async def test_capture_with_complex_headers(self, mock_request):
        """Test capturing request with complex headers."""
        # Arrange
        json_data = {"data": "test"}
        json_bytes = json.dumps(json_data).encode("utf-8")
        
        async def mock_body():
            return json_bytes
        
        mock_request.body = mock_body
        mock_request.headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": "Bearer token123",
            "X-Custom-Header": "custom-value",
            "User-Agent": "TestClient/1.0"
        }

        # Act
        response = await capture_raw_request(mock_request)

        # Assert
        assert response["headers"]["Content-Type"] == "application/json; charset=utf-8"
        assert response["headers"]["Authorization"] == "Bearer token123"
        assert response["headers"]["X-Custom-Header"] == "custom-value"
        assert response["headers"]["User-Agent"] == "TestClient/1.0"

    @pytest.mark.asyncio
    async def test_capture_nested_json_structure(self, mock_request):
        """Test capturing request with nested JSON structure."""
        # Arrange
        json_data = {
            "user": {
                "name": "John Doe",
                "email": "john@example.com",
                "metadata": {
                    "created": "2023-01-01",
                    "tags": ["user", "active"]
                }
            }
        }
        json_bytes = json.dumps(json_data).encode("utf-8")
        
        async def mock_body():
            return json_bytes
        
        mock_request.body = mock_body

        # Act
        response = await capture_raw_request(mock_request)

        # Assert
        assert response["body_info"]["json_valid"] is True
        assert response["body_info"]["json_data"] == json_data
        assert response["body_info"]["json_data"]["user"]["name"] == "John Doe"
        assert response["body_info"]["json_data"]["user"]["metadata"]["tags"] == ["user", "active"]

    @pytest.mark.asyncio
    async def test_capture_url_encoded_data(self, mock_request):
        """Test capturing URL-encoded form data."""
        # Arrange
        form_data = "username=testuser&password=testpass&remember=true"
        form_bytes = form_data.encode("utf-8")
        
        async def mock_body():
            return form_bytes
        
        mock_request.body = mock_body
        mock_request.headers = {"Content-Type": "application/x-www-form-urlencoded"}

        # Act
        response = await capture_raw_request(mock_request)

        # Assert
        assert response["body_info"]["decoded_success"] is True
        assert response["body_info"]["json_valid"] is False
        assert response["body_info"]["decoded_string"] == form_data

    @pytest.mark.asyncio
    async def test_capture_binary_data(self, mock_request):
        """Test capturing binary data."""
        # Arrange
        test_data = b'\x00\x01\x02\x03\x04\x05'
        
        async def mock_body():
            return test_data
        
        mock_request.body = mock_body
        mock_request.headers = {"Content-Type": "application/octet-stream"}

        # Act
        response = await capture_raw_request(mock_request)

        # Assert
        assert response["body_info"]["length_bytes"] == len(test_data)
        # This data is actually valid UTF-8 (null bytes and control chars are valid UTF-8)
        # But it's not valid JSON, so decoding succeeds but JSON parsing fails
        assert response["body_info"]["decoded_success"] is True
        assert response["body_info"]["json_valid"] is False
        assert "raw_bytes" in response["body_info"]

    @pytest.mark.asyncio
    async def test_capture_large_json_payload(self, mock_request):
        """Test capturing large JSON payload."""
        # Arrange
        large_data = {"items": [f"item_{i}" for i in range(1000)]}
        json_bytes = json.dumps(large_data).encode("utf-8")
        
        async def mock_body():
            return json_bytes
        
        mock_request.body = mock_body
        mock_request.headers = {"Content-Type": "application/json"}

        # Act
        response = await capture_raw_request(mock_request)

        # Assert
        assert response["body_info"]["json_valid"] is True
        assert len(response["body_info"]["json_data"]["items"]) == 1000
        assert response["body_info"]["json_data"]["items"][0] == "item_0"
        assert response["body_info"]["json_data"]["items"][999] == "item_999"

    @pytest.mark.asyncio
    async def test_capture_json_with_special_characters(self, mock_request):
        """Test capturing JSON with special characters."""
        # Arrange
        json_data = {
            "message": "Hello, 世界! 🌍",
            "symbols": "©®™€£¥",
            "unicode": "\\u0041\\u0042\\u0043"
        }
        json_bytes = json.dumps(json_data).encode("utf-8")
        
        async def mock_body():
            return json_bytes
        
        mock_request.body = mock_body

        # Act
        response = await capture_raw_request(mock_request)

        # Assert
        assert response["body_info"]["json_valid"] is True
        assert response["body_info"]["json_data"]["message"] == "Hello, 世界! 🌍"
        assert response["body_info"]["json_data"]["symbols"] == "©®™€£¥"

    @pytest.mark.asyncio
    async def test_capture_malformed_utf8_sequences(self, mock_request):
        """Test capturing request with malformed UTF-8 sequences."""
        # Arrange
        invalid_utf8 = b'\xc0\xaf\xed\xa0\x80\xed\xbf\xbf'  # Invalid UTF-8 sequences
        
        async def mock_body():
            return invalid_utf8
        
        mock_request.body = mock_body

        # Act
        response = await capture_raw_request(mock_request)

        # Assert
        assert response["body_info"]["decoded_success"] is False
        assert "decode_error" in response["body_info"]
        assert "raw_bytes" in response["body_info"]

    @pytest.mark.asyncio
    @patch('clarity.api.v1.debug.logger')
    async def test_capture_logs_errors(self, mock_logger, mock_request):
        """Test that capture endpoint logs errors properly."""
        # Arrange
        async def mock_body():
            raise Exception("Test error")
        
        mock_request.body = mock_body

        # Act
        response = await capture_raw_request(mock_request)

        # Assert
        assert response["error"] == "Test error"
        assert response["type"] == "Exception"
        mock_logger.exception.assert_called_once_with("Debug endpoint error")


class TestEchoLoginRequestEndpoint:
    """Test the /debug/echo-login endpoint."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock request object."""
        request = MagicMock(spec=Request)
        request.method = "POST"
        request.url = "http://test.com/debug/echo-login"
        request.headers = {"Content-Type": "application/json"}
        return request

    @pytest.mark.asyncio
    async def test_echo_valid_login_request(self, mock_request):
        """Test echoing a valid login request."""
        # Arrange
        login_data = {"username": "testuser", "password": "testpass"}
        login_bytes = json.dumps(login_data).encode("utf-8")
        
        async def mock_body():
            return login_bytes
        
        mock_request.body = mock_body

        # Act
        response = await echo_login_request(mock_request)

        # Assert
        assert isinstance(response, JSONResponse)
        # Parse the response content
        response_data = json.loads(response.body)
        assert response_data["access_token"] == "debug_token"
        assert response_data["refresh_token"] == "debug_refresh"
        assert response_data["token_type"] == "bearer"
        assert response_data["expires_in"] == 3600
        assert response_data["scope"] == "full_access"

    @pytest.mark.asyncio
    async def test_echo_empty_request(self, mock_request):
        """Test echoing an empty request."""
        # Arrange
        async def mock_body():
            return b""
        
        mock_request.body = mock_body

        # Act
        response = await echo_login_request(mock_request)

        # Assert
        assert isinstance(response, JSONResponse)
        response_data = json.loads(response.body)
        assert response_data["access_token"] == "debug_token"
        assert response_data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_echo_invalid_utf8_request(self, mock_request):
        """Test echoing a request with invalid UTF-8."""
        # Arrange
        invalid_utf8 = b'\x80\x81\x82\x83'
        
        async def mock_body():
            return invalid_utf8
        
        mock_request.body = mock_body

        # Act
        response = await echo_login_request(mock_request)

        # Assert
        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        response_data = json.loads(response.body)
        assert "error" in response_data

    @pytest.mark.asyncio
    async def test_echo_with_request_details(self, mock_request):
        """Test echoing with full request details."""
        # Arrange
        login_data = {"username": "admin", "password": "secret"}
        login_bytes = json.dumps(login_data).encode("utf-8")
        
        async def mock_body():
            return login_bytes
        
        mock_request.body = mock_body
        mock_request.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer token123",
            "X-Request-ID": "req-123"
        }

        # Act
        response = await echo_login_request(mock_request)

        # Assert
        assert isinstance(response, JSONResponse)
        response_data = json.loads(response.body)
        assert response_data["access_token"] == "debug_token"
        assert response_data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_echo_complex_login_data(self, mock_request):
        """Test echoing complex login data structure."""
        # Arrange
        login_data = {
            "username": "testuser",
            "password": "testpass",
            "remember_me": True,
            "metadata": {
                "device": "mobile",
                "app_version": "1.0.0",
                "platform": "iOS"
            }
        }
        login_bytes = json.dumps(login_data).encode("utf-8")
        
        async def mock_body():
            return login_bytes
        
        mock_request.body = mock_body

        # Act
        response = await echo_login_request(mock_request)

        # Assert
        assert isinstance(response, JSONResponse)
        response_data = json.loads(response.body)
        assert response_data["access_token"] == "debug_token"
        assert response_data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_echo_special_characters_in_login(self, mock_request):
        """Test echoing login data with special characters."""
        # Arrange
        login_data = {
            "username": "user@example.com",
            "password": "P@ssw0rd!#$%^&*()",
            "display_name": "John Doe 🎉"
        }
        login_bytes = json.dumps(login_data).encode("utf-8")
        
        async def mock_body():
            return login_bytes
        
        mock_request.body = mock_body

        # Act
        response = await echo_login_request(mock_request)

        # Assert
        assert isinstance(response, JSONResponse)
        response_data = json.loads(response.body)
        assert response_data["access_token"] == "debug_token"
        assert response_data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_echo_large_login_payload(self, mock_request):
        """Test echoing large login payload."""
        # Arrange
        login_data = {
            "username": "testuser",
            "password": "testpass",
            "extra_data": "x" * 1000  # Large string
        }
        login_bytes = json.dumps(login_data).encode("utf-8")
        
        async def mock_body():
            return login_bytes
        
        mock_request.body = mock_body

        # Act
        response = await echo_login_request(mock_request)

        # Assert
        assert isinstance(response, JSONResponse)
        response_data = json.loads(response.body)
        assert response_data["access_token"] == "debug_token"
        assert response_data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_echo_binary_data_handling(self, mock_request):
        """Test echo endpoint handling binary data."""
        # Arrange
        binary_data = b'\x00\x01\x02\x03'
        
        async def mock_body():
            return binary_data
        
        mock_request.body = mock_body

        # Act
        response = await echo_login_request(mock_request)

        # Assert
        assert isinstance(response, JSONResponse)
        # Binary data with null bytes is valid UTF-8 and decodes to empty string
        # So no error should be raised - the implementation correctly handles it
        assert response.status_code == 200
        response_data = json.loads(response.body)
        assert response_data["access_token"] == "debug_token"

    @pytest.mark.asyncio
    @patch('clarity.api.v1.debug.logger')
    async def test_echo_logs_requests(self, mock_logger, mock_request):
        """Test that echo endpoint logs requests properly."""
        # Arrange
        login_data = {"username": "testuser", "password": "testpass"}
        login_bytes = json.dumps(login_data).encode("utf-8")
        
        async def mock_body():
            return login_bytes
        
        mock_request.body = mock_body

        # Act
        response = await echo_login_request(mock_request)

        # Assert
        assert isinstance(response, JSONResponse)
        response_data = json.loads(response.body)
        assert response_data["access_token"] == "debug_token"
        # Verify logging was called
        mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_echo_with_null_bytes(self, mock_request):
        """Test echo endpoint with null bytes in data."""
        # Arrange
        data_with_nulls = b'test\x00data\x00with\x00nulls'
        
        async def mock_body():
            return data_with_nulls
        
        mock_request.body = mock_body

        # Act
        response = await echo_login_request(mock_request)

        # Assert
        assert isinstance(response, JSONResponse)
        # The actual implementation should handle this gracefully
        response_data = json.loads(response.body)
        assert response_data["access_token"] == "debug_token"

    @pytest.mark.asyncio
    async def test_echo_extremely_long_string(self, mock_request):
        """Test echo endpoint with extremely long string."""
        # Arrange
        long_string = "a" * 10000  # Very long string
        long_bytes = long_string.encode("utf-8")
        
        async def mock_body():
            return long_bytes
        
        mock_request.body = mock_body

        # Act
        response = await echo_login_request(mock_request)

        # Assert
        assert isinstance(response, JSONResponse)
        response_data = json.loads(response.body)
        assert response_data["access_token"] == "debug_token"

    @pytest.mark.asyncio
    async def test_echo_json_parse_error_at_start(self, mock_request):
        """Test echo endpoint with JSON parse error at start."""
        # Arrange
        error_at_start = '{invalid json'
        
        async def mock_body():
            return error_at_start.encode("utf-8")
        
        mock_request.body = mock_body

        # Act
        response = await echo_login_request(mock_request)

        # Assert
        assert isinstance(response, JSONResponse)
        response_data = json.loads(response.body)
        assert response_data["access_token"] == "debug_token" 