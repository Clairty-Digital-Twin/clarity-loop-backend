"""Test version module."""

from unittest.mock import patch
import pytest
from clarity.version import get_version


def test_get_version_installed():
    """Test get_version when package is installed."""
    with patch("clarity.version.metadata.version") as mock_version:
        mock_version.return_value = "1.2.3"
        assert get_version() == "1.2.3"
        mock_version.assert_called_once_with("clarity-loop-backend")


def test_get_version_not_installed():
    """Test get_version when package is not installed."""
    with patch("clarity.version.metadata.version") as mock_version:
        # Simulate PackageNotFoundError
        from importlib import metadata
        mock_version.side_effect = metadata.PackageNotFoundError("clarity-loop-backend")
        
        assert get_version() == "0.1.0-dev"