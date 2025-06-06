from unittest.mock import MagicMock, patch

import pytest

from clarity.auth import firebase_auth


def test_initialize_firebase_with_credentials_path():
    """Test that Firebase is initialized with a credentials path if provided.
    """
    mock_settings = MagicMock()
    mock_settings.firebase_credentials_path = "fake/path/to/creds.json"

    with patch.object(firebase_auth, "settings", mock_settings), \
         patch.object(firebase_auth.credentials, "Certificate") as mock_creds, \
         patch.object(firebase_auth.firebase_admin, "initialize_app") as mock_init, \
         patch.object(firebase_auth.firebase_admin, "get_app", side_effect=ValueError):

        # Reload the module to trigger initialization
        import importlib
        importlib.reload(firebase_auth)

        mock_creds.assert_called_once_with("fake/path/to/creds.json")
        mock_init.assert_called_once()
