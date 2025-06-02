"""CLARITY Digital Twin Platform - Configuration Management.

Environment-based configuration using Pydantic settings for secure,
production-ready deployment across development, staging, and production.
"""

from functools import lru_cache
import logging
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

# Configure logger
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application configuration with environment variable support."""

    # Application settings
    app_name: str = "CLARITY Digital Twin Platform"
    app_version: str = "1.0.0"
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")

    # Server settings
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8080, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Security settings
    allowed_hosts: list[str] = Field(
        default=["localhost", "127.0.0.1", "*.run.app"], alias="ALLOWED_HOSTS"
    )
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"], alias="CORS_ORIGINS"
    )

    # Authentication settings
    enable_auth: bool = Field(default=False, alias="ENABLE_AUTH")

    # Firebase settings
    firebase_project_id: str = Field(default="", alias="FIREBASE_PROJECT_ID")
    firebase_credentials_path: str = Field(
        default="", alias="FIREBASE_CREDENTIALS_PATH"
    )

    # Google Cloud settings
    gcp_project_id: str = Field(default="", alias="GCP_PROJECT_ID")
    firestore_database: str = Field(default="(default)", alias="FIRESTORE_DATABASE")
    google_application_credentials: str = Field(
        default="", alias="GOOGLE_APPLICATION_CREDENTIALS"
    )

    # Vertex AI settings
    vertex_ai_project_id: str = Field(default="", alias="VERTEX_AI_PROJECT_ID")
    vertex_ai_location: str = Field(default="us-central1", alias="VERTEX_AI_LOCATION")
    vertex_ai_model_id: str = Field(
        default="gemini-2.5-pro-preview-05-06", alias="VERTEX_AI_MODEL_ID"
    )

    # Storage settings
    cloud_storage_bucket: str = Field(default="", alias="CLOUD_STORAGE_BUCKET")

    # Development overrides to prevent startup hangs
    skip_external_services: bool | None = Field(
        default=None, alias="SKIP_EXTERNAL_SERVICES"
    )
    startup_timeout: float = Field(default=15.0, alias="STARTUP_TIMEOUT")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    @model_validator(mode="after")
    def validate_environment_requirements(self) -> Self:
        """Validate environment-specific requirements and set development defaults."""
        # Set development defaults for skip_external_services
        if self.skip_external_services is None:
            self.skip_external_services = self.environment.lower() == "development"

        # In development, warn about missing credentials but don't fail
        if self.environment.lower() == "development":
            missing_creds: list[str] = []

            if not self.firebase_project_id:
                missing_creds.append("FIREBASE_PROJECT_ID")
            if (
                not self.firebase_credentials_path
                and not self.google_application_credentials
            ):
                missing_creds.append(
                    "FIREBASE_CREDENTIALS_PATH or GOOGLE_APPLICATION_CREDENTIALS"
                )
            if not self.gcp_project_id:
                missing_creds.append("GCP_PROJECT_ID")

            if missing_creds:
                logger.warning(
                    f"⚠️ Development mode: Missing credentials {missing_creds}. "
                    f"Using mock services (skip_external_services={self.skip_external_services})"
                )

        # In production, require critical credentials
        elif self.environment.lower() == "production":
            required_for_production: list[str] = []

            if self.enable_auth and not self.firebase_project_id:
                required_for_production.append("FIREBASE_PROJECT_ID (auth enabled)")

            if self.enable_auth and not (
                self.firebase_credentials_path or self.google_application_credentials
            ):
                required_for_production.append(
                    "FIREBASE_CREDENTIALS_PATH or GOOGLE_APPLICATION_CREDENTIALS (auth enabled)"
                )

            if not self.skip_external_services and not self.gcp_project_id:
                required_for_production.append(
                    "GCP_PROJECT_ID (external services enabled)"
                )

            if required_for_production:
                missing_vars = ", ".join(required_for_production)
                msg = (
                    f"Production environment requires: {missing_vars}. "
                    f"Set SKIP_EXTERNAL_SERVICES=true to use mock services."
                )
                raise ValueError(
                    msg
                )

        return self

    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"

    def should_use_mock_services(self) -> bool:
        """Check if mock services should be used instead of external services."""
        return self.skip_external_services or self.is_development()

    def get_startup_timeout(self) -> float:
        """Get the startup timeout in seconds."""
        return self.startup_timeout

    def log_configuration_summary(self) -> None:
        """Log configuration summary for debugging."""
        logger.info("🔧 CLARITY Configuration Summary:")
        logger.info(f"   • Environment: {self.environment}")
        logger.info(f"   • Debug mode: {self.debug}")
        logger.info(f"   • Auth enabled: {self.enable_auth}")
        logger.info(f"   • Skip external services: {self.skip_external_services}")
        logger.info(f"   • Startup timeout: {self.startup_timeout}s")
        logger.info(f"   • Firebase project: {self.firebase_project_id or 'Not set'}")
        logger.info(f"   • GCP project: {self.gcp_project_id or 'Not set'}")


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    settings = Settings()

    # Log configuration summary in debug mode
    if settings.debug or settings.log_level.upper() == "DEBUG":
        settings.log_configuration_summary()

    return settings
