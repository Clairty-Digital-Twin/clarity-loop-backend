"""Test container module."""

from unittest.mock import Mock, patch
import pytest
from clarity.core.container import create_application


def test_create_application():
    """Test create_application returns FastAPI app."""
    mock_app = Mock()
    
    with patch("clarity.core.container.clarity_app", mock_app):
        result = create_application()
        assert result == mock_app