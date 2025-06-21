"""Tests for GCP credentials management."""

from collections.abc import Generator
import json
import os
from pathlib import Path
from typing import Any

import pytest

from clarity.services.gcp_credentials import (
    GCPCredentialsManager,
    get_gcp_credentials_manager,
)


class TestGCPCredentialsManager:
    """Test cases for GCP credentials manager."""

    @pytest.fixture
    def mock_credentials(self) -> dict[str, Any]:
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
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/test%40test-project.iam.gserviceaccount.com",
        }

    @pytest.fixture
    def clean_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> Generator[None, None, None]:
        """Clean up environment variables before and after tests."""
        # Store original values
        original_gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        original_gac_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")

        # Clean environment
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", raising=False)

        # Reset the singleton instance
        GCPCredentialsManager._instance = None

        yield

        # Cleanup any temporary files created during tests
        manager = GCPCredentialsManager()
        if (
            hasattr(manager, "_credentials_file_path")
            and manager._credentials_file_path
        ):
            try:
                if Path(manager._credentials_file_path).exists():
                    Path(manager._credentials_file_path).unlink()
            except Exception:  # noqa: BLE001, S110
                pass

        # Restore original values if they existed
        if original_gac:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = original_gac
        if original_gac_json:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"] = original_gac_json

        # Reset singleton again
        GCPCredentialsManager._instance = None

    def test_singleton_pattern(self, clean_environment: Any) -> None:
        """Test that GCPCredentialsManager follows singleton pattern."""
        manager1 = GCPCredentialsManager()
        manager2 = GCPCredentialsManager()
        assert manager1 is manager2

    def test_credentials_from_env_json(
        self, clean_environment: Any, mock_credentials: dict[str, Any]
    ) -> None:
        """Test loading credentials from GOOGLE_APPLICATION_CREDENTIALS_JSON."""
        # Set the JSON in environment
        os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"] = json.dumps(mock_credentials)

        # Create manager instance
        GCPCredentialsManager()

        # Check that credentials file was created
        assert os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is not None
        credentials_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        assert Path(credentials_path).exists()

        # Verify the content
        with Path(credentials_path).open(encoding="utf-8") as f:
            loaded_creds = json.load(f)
            assert loaded_creds == mock_credentials

    def test_local_credentials_file(self, clean_environment: Any) -> None:
        """Test loading credentials from local file."""
        # Create a local credentials file
        local_file = Path("gcp-service-account.json")
        test_creds = {"type": "service_account", "project_id": "local-test"}

        try:
            with local_file.open("w", encoding="utf-8") as f:
                json.dump(test_creds, f)

            # Create manager instance
            GCPCredentialsManager()

            # Check that it found the local file
            assert os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") == str(
                local_file.absolute()
            )
        finally:
            # Clean up
            if local_file.exists():
                local_file.unlink()

    def test_existing_google_credentials(
        self, clean_environment: Any, tmp_path: Path, mock_credentials: dict[str, Any]
    ) -> None:
        """Test when GOOGLE_APPLICATION_CREDENTIALS is already set."""
        # Create a credentials file
        credentials_file = tmp_path / "existing-creds.json"
        with credentials_file.open("w", encoding="utf-8") as f:
            json.dump(mock_credentials, f)

        # Set the environment variable
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_file)

        # Change to temp directory
        original_cwd = Path.cwd()
        os.chdir(tmp_path)

        try:
            # Create manager instance
            manager = GCPCredentialsManager()

            # It should use the existing credentials
            assert manager.get_credentials_path() == str(credentials_file)
        finally:
            os.chdir(original_cwd)

    def test_no_credentials_warning(self, clean_environment: Any, caplog: Any) -> None:
        """Test warning when no credentials are found."""
        manager = GCPCredentialsManager()
        assert manager.get_credentials_path() is None
        assert "No GCP credentials found" in caplog.text

    def test_get_project_id(
        self, clean_environment: Any, tmp_path: Path, mock_credentials: dict[str, Any]
    ) -> None:
        """Test extracting project ID from credentials."""
        # Create a temporary credentials file
        credentials_file = tmp_path / "credentials.json"
        with credentials_file.open("w", encoding="utf-8") as f:
            json.dump(mock_credentials, f)

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_file)

        manager = GCPCredentialsManager()
        assert manager.get_project_id() == "test-project"

    def test_get_project_id_no_credentials(self, clean_environment: Any) -> None:
        """Test get_project_id when no credentials are set."""
        manager = GCPCredentialsManager()
        assert manager.get_project_id() is None

    def test_cleanup_temp_file(
        self, clean_environment: Any, mock_credentials: dict[str, Any]
    ) -> None:
        """Test cleanup of temporary credentials file."""
        os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"] = json.dumps(mock_credentials)

        manager = GCPCredentialsManager()
        temp_file_path = manager._credentials_file_path

        assert Path(temp_file_path).exists()

        manager.cleanup()

        assert not Path(temp_file_path).exists()

    def test_get_gcp_credentials_manager(self) -> None:
        """Test getting the global credentials manager instance."""
        manager1 = get_gcp_credentials_manager()
        manager2 = get_gcp_credentials_manager()
        assert manager1 is manager2
