import importlib
from unittest.mock import MagicMock, patch

from clarity.auth import firebase_auth


def test_initialize_firebase_with_credentials_path():
    """Test that Firebase is initialized with a credentials path if provided."""
    mock_settings = MagicMock()
    mock_settings.firebase_credentials_path = "fake/path/to/creds.json"

    # Mock the credentials Certificate class before reloading
    with (
        patch("firebase_admin.credentials.Certificate") as mock_creds,
        patch("firebase_admin.initialize_app") as mock_init,
        patch("firebase_admin.get_app", side_effect=ValueError),
        patch("clarity.core.config.get_settings", return_value=mock_settings),
    ):

        # Reload the module to trigger initialization
        importlib.reload(firebase_auth)

        mock_creds.assert_called_once_with("fake/path/to/creds.json")
        mock_init.assert_called_once()
