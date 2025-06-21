"""Google Cloud Platform credentials management.

This module handles loading GCP service account credentials from environment
variables when running in containerized environments like ECS.
"""

import json
import logging
import os
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


class GCPCredentialsManager:
    """Manages Google Cloud Platform credentials for the application."""

    _instance = None
    _credentials_file_path: str | None = None

    def __new__(cls) -> "GCPCredentialsManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the GCP credentials manager."""
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._setup_credentials()

    def _setup_credentials(self) -> None:
        """Set up GCP credentials from environment variables or local files."""
        # Check if credentials are already set via environment
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            logger.info("Using existing GOOGLE_APPLICATION_CREDENTIALS")
            return

        # Check for credentials JSON in environment variable (from AWS Secrets Manager)
        credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if credentials_json:
            logger.info(
                "Found GOOGLE_APPLICATION_CREDENTIALS_JSON, creating temporary file"
            )
            self._create_temp_credentials_file(credentials_json)
            return

        # Check for local credentials files (development)
        local_files = [
            "gcp-service-account.json",
            "service-account.json",
            "gcp-credentials.json",
        ]

        for filename in local_files:
            if Path(filename).exists():
                logger.info("Using local service account file: %s", filename)
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
                    Path(filename).absolute()
                )
                return

        logger.warning("No GCP credentials found. Some features may not work.")

    def _create_temp_credentials_file(self, credentials_json: str) -> None:
        """Create a temporary file with the GCP credentials.

        Args:
            credentials_json: JSON string containing the service account credentials
        """
        try:
            # Parse JSON to validate it
            credentials_data = json.loads(credentials_json)

            # Create a temporary file that won't be deleted on close
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix="gcp_credentials_",
                delete=False,
                encoding="utf-8",
            ) as temp_file:
                json.dump(credentials_data, temp_file, indent=2)
                self._credentials_file_path = temp_file.name

            # Set the environment variable to point to the temp file
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self._credentials_file_path
            logger.info(
                "Created temporary credentials file at: %s", self._credentials_file_path
            )

        except json.JSONDecodeError:
            logger.exception("Invalid JSON in GOOGLE_APPLICATION_CREDENTIALS_JSON")
            raise
        except Exception:
            logger.exception("Error creating temporary credentials file")
            raise

    def get_credentials_path(self) -> str | None:
        """Get the path to the GCP credentials file.

        Returns:
            Path to the credentials file or None if not set
        """
        return os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    def get_project_id(self) -> str | None:
        """Get the GCP project ID from the credentials.

        Returns:
            Project ID or None if not available
        """
        credentials_path = self.get_credentials_path()
        if not credentials_path:
            return None

        try:
            with Path(credentials_path).open(encoding="utf-8") as f:
                credentials_data = json.load(f)
                project_id = credentials_data.get("project_id")
                return project_id if isinstance(project_id, str) else None
        except Exception:
            logger.exception("Error reading project ID from credentials")
            return None

    def cleanup(self) -> None:
        """Clean up temporary credentials file if created."""
        if self._credentials_file_path and Path(self._credentials_file_path).exists():
            try:
                Path(self._credentials_file_path).unlink()
                logger.info(
                    "Cleaned up temporary credentials file: %s",
                    self._credentials_file_path,
                )
            except Exception:
                logger.exception("Error cleaning up credentials file")


# Global instance
_credentials_manager = GCPCredentialsManager()


def get_gcp_credentials_manager() -> GCPCredentialsManager:
    """Get the global GCP credentials manager instance."""
    return _credentials_manager


def initialize_gcp_credentials() -> None:
    """Initialize GCP credentials at application startup.

    This should be called early in the application lifecycle.
    """
    manager = get_gcp_credentials_manager()
    credentials_path = manager.get_credentials_path()

    if credentials_path:
        logger.info("GCP credentials initialized: %s", credentials_path)
        project_id = manager.get_project_id()
        if project_id:
            logger.info("GCP project ID: %s", project_id)
    else:
        logger.warning("GCP credentials not configured")
