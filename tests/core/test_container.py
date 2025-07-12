"""Test container module."""

from unittest.mock import Mock, patch



def test_create_application():
    """Test create_application returns FastAPI app."""
    # Mock the main app to avoid full initialization
    mock_app = Mock()
    mock_app.title = "Clarity Loop Backend"

    with patch("clarity.main.app", mock_app):
        from clarity.core.container import create_application

        result = create_application()

        assert result == mock_app
        assert result.title == "Clarity Loop Backend"
