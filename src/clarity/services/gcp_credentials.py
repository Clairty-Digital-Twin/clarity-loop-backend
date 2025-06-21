"""
Google Cloud Platform credentials management.

This module handles loading GCP service account credentials from environment
variables when running in containerized environments like ECS.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class GCPCredentialsManager:
    """Manages Google Cloud Platform credentials for the application."""
    
    _instance = None
    _credentials_file_path: Optional[str] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the GCP credentials manager."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._setup_credentials()
    
    def _setup_credentials(self) -> None:
        """Setup GCP credentials from environment variable or file."""
        # First check if GOOGLE_APPLICATION_CREDENTIALS is already set
        if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            logger.info("Using existing GOOGLE_APPLICATION_CREDENTIALS path")
            return
        
        # Check for credentials JSON in environment variable (from AWS Secrets Manager)
        credentials_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
        if credentials_json:
            logger.info("Found GCP credentials in environment variable")
            self._create_temp_credentials_file(credentials_json)
        else:
            # Look for local service account file (development)
            local_files = [
                'clarity-loop-backend-f770782498c7.json',
                'service-account.json',
                'gcp-credentials.json'
            ]
            
            for filename in local_files:
                if Path(filename).exists():
                    logger.info(f"Using local service account file: {filename}")
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(Path(filename).absolute())
                    return
            
            logger.warning("No GCP credentials found. Some features may not work.")
    
    def _create_temp_credentials_file(self, credentials_json: str) -> None:
        """
        Create a temporary file with the GCP credentials.
        
        Args:
            credentials_json: JSON string containing the service account credentials
        """
        try:
            # Parse JSON to validate it
            credentials_data = json.loads(credentials_json)
            
            # Create a temporary file that won't be deleted on close
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.json',
                prefix='gcp_credentials_',
                delete=False
            ) as temp_file:
                json.dump(credentials_data, temp_file, indent=2)
                self._credentials_file_path = temp_file.name
            
            # Set the environment variable to point to the temp file
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self._credentials_file_path
            logger.info(f"Created temporary credentials file at: {self._credentials_file_path}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating temporary credentials file: {e}")
            raise
    
    def get_credentials_path(self) -> Optional[str]:
        """
        Get the path to the GCP credentials file.
        
        Returns:
            Path to the credentials file or None if not set
        """
        return os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    
    def get_project_id(self) -> Optional[str]:
        """
        Get the GCP project ID from the credentials.
        
        Returns:
            Project ID or None if not available
        """
        credentials_path = self.get_credentials_path()
        if not credentials_path:
            return None
        
        try:
            with open(credentials_path, 'r') as f:
                credentials_data = json.load(f)
                return credentials_data.get('project_id')
        except Exception as e:
            logger.error(f"Error reading project ID from credentials: {e}")
            return None
    
    def cleanup(self) -> None:
        """Clean up temporary credentials file if created."""
        if self._credentials_file_path and Path(self._credentials_file_path).exists():
            try:
                os.unlink(self._credentials_file_path)
                logger.info(f"Cleaned up temporary credentials file: {self._credentials_file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up credentials file: {e}")


# Global instance
_credentials_manager = GCPCredentialsManager()


def get_gcp_credentials_manager() -> GCPCredentialsManager:
    """Get the global GCP credentials manager instance."""
    return _credentials_manager


def initialize_gcp_credentials() -> None:
    """
    Initialize GCP credentials at application startup.
    This should be called early in the application lifecycle.
    """
    manager = get_gcp_credentials_manager()
    credentials_path = manager.get_credentials_path()
    
    if credentials_path:
        logger.info(f"GCP credentials initialized: {credentials_path}")
        project_id = manager.get_project_id()
        if project_id:
            logger.info(f"GCP project ID: {project_id}")
    else:
        logger.warning("GCP credentials not configured")