"""Test middleware ports module."""

import pytest
from unittest.mock import AsyncMock, Mock
from fastapi import Request, Response

from clarity.ports.middleware_ports import IMiddleware


class MockMiddleware(IMiddleware):
    """Mock implementation of IMiddleware for testing."""
    
    async def __call__(self, request: Request, call_next):
        # Simple implementation that adds a header
        response = await call_next(request)
        response.headers["X-Mock-Middleware"] = "processed"
        return response


def test_imiddleware_abstract():
    """Test that IMiddleware is abstract."""
    with pytest.raises(TypeError):
        IMiddleware()


@pytest.mark.asyncio
async def test_mock_middleware_implementation():
    """Test mock middleware implementation."""
    middleware = MockMiddleware()
    
    # Create mock request and response
    mock_request = Mock(spec=Request)
    mock_response = Mock(spec=Response)
    mock_response.headers = {}
    
    # Create mock call_next that returns the response
    async def mock_call_next(request):
        return mock_response
    
    # Process request through middleware
    result = await middleware(mock_request, mock_call_next)
    
    assert result == mock_response
    assert result.headers["X-Mock-Middleware"] == "processed"