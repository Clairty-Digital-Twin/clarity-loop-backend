"""Tests for GCP credentials management."""

import json
import os
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from clarity.services.gcp_credentials import (
    GCPCredentialsManager,
    get_gcp_credentials_manager,
    initialize_gcp_credentials,
)


class TestGCPCredentialsManager:
    """Test cases for GCP credentials manager."""

    @pytest.fixture
    def mock_credentials(self):
        """Sample GCP service account credentials."""
        return {
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "test-key-id",
            "private_key": "-----BEGIN PRIVATE KEY-----\ntest-key\n-----END PRIVATE KEY-----\n",
            "client_email": "test@test-project.iam.gserviceaccount.com",
            "client_id": "123456789",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/test%40test-project.iam.gserviceaccount.com"
        }

    @pytest.fixture
    def clean_environment(self, monkeypatch):
        """Clean up environment variables before and after tests."""
        # Store original values
        original_gac = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        original_gac_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')

        # Clean environment
        monkeypatch.delenv('GOOGLE_APPLICATION_CREDENTIALS', raising=False)
        monkeypatch.delenv('GOOGLE_APPLICATION_CREDENTIALS_JSON', raising=False)

        # Reset the singleton instance
        GCPCredentialsManager._instance = None

        yield

        # Restore original values
        if original_gac:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = original_gac
        else:
            os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS', None)

        if original_gac_json:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'] = original_gac_json
        else:
            os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS_JSON', None)

        # Reset the singleton instance
        GCPCredentialsManager._instance = None

    def test_singleton_pattern(self):
        """Test that GCPCredentialsManager follows singleton pattern."""
        # Reset singleton first
        GCPCredentialsManager._instance = None

        manager1 = GCPCredentialsManager()
        manager2 = GCPCredentialsManager()
        assert manager1 is manager2

    def test_existing_credentials_path(self, clean_environment):
        """Test when GOOGLE_APPLICATION_CREDENTIALS is already set."""
        test_path = "/path/to/credentials.json"
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = test_path

        manager = GCPCredentialsManager()
        assert manager.get_credentials_path() == test_path

    def test_credentials_from_json_env(self, clean_environment, mock_credentials):
        """Test loading credentials from JSON environment variable."""
        os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'] = json.dumps(mock_credentials)

        manager = GCPCredentialsManager()
        credentials_path = manager.get_credentials_path()

        assert credentials_path is not None
        assert Path(credentials_path).exists()

        # Verify file contents
        with open(credentials_path) as f:
            loaded_credentials = json.load(f)
        assert loaded_credentials == mock_credentials

        # Clean up
        manager.cleanup()

    def test_invalid_json_credentials(self, clean_environment):
        """Test handling of invalid JSON in credentials."""
        os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'] = "invalid json"

        with pytest.raises(json.JSONDecodeError):
            GCPCredentialsManager()

    def test_local_file_fallback(self, clean_environment, tmp_path, mock_credentials):
        """Test fallback to local service account file."""
        # Create a temporary service account file
        service_account_file = tmp_path / "service-account.json"
        with open(service_account_file, 'w') as f:
            json.dump(mock_credentials, f)

        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            manager = GCPCredentialsManager()
            assert manager.get_credentials_path() == str(service_account_file.absolute())
        finally:
            os.chdir(original_cwd)

    def test_no_credentials_warning(self, clean_environment, caplog):
        """Test warning when no credentials are found."""
        manager = GCPCredentialsManager()
        assert manager.get_credentials_path() is None
        assert "No GCP credentials found" in caplog.text

    def test_get_project_id(self, clean_environment, tmp_path, mock_credentials):
        """Test extracting project ID from credentials."""
        # Create a temporary credentials file
        credentials_file = tmp_path / "credentials.json"
        with open(credentials_file, 'w') as f:
            json.dump(mock_credentials, f)

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(credentials_file)

        manager = GCPCredentialsManager()
        assert manager.get_project_id() == "test-project"

    def test_get_project_id_no_credentials(self, clean_environment):
        """Test get_project_id when no credentials are set."""
        manager = GCPCredentialsManager()
        assert manager.get_project_id() is None

    def test_cleanup_temp_file(self, clean_environment, mock_credentials):
        """Test cleanup of temporary credentials file."""
        os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'] = json.dumps(mock_credentials)

        manager = GCPCredentialsManager()
        temp_file_path = manager._credentials_file_path

        assert Path(temp_file_path).exists()

        manager.cleanup()

        assert not Path(temp_file_path).exists()

    def test_get_gcp_credentials_manager(self):
        """Test getting the global credentials manager instance."""
        manager1 = get_gcp_credentials_manager()
        manager2 = get_gcp_credentials_manager()
        assert manager1 is manager2
