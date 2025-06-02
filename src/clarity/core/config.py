"""CLARITY Digital Twin Platform - Configuration Management.

Environment-based configuration using Pydantic settings for secure,
production-ready deployment across development, staging, and production.
"""

from dataclasses import dataclass
from functools import lru_cache
import logging
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class MiddlewareConfig:
    """Configuration for Firebase authentication middleware.

    Contains all middleware-specific settings for authentication,
    token caching, and error handling.
    """

    # Authentication settings
    enabled: bool = True

    # Exempt paths (no authentication required)
    exempt_paths: list[str] = None

    # Token cache settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    cache_max_size: int = 1000

    # Error handling settings
    graceful_degradation: bool = True
    fallback_to_mock: bool = True
    initialization_timeout_seconds: int = 8

    # Logging settings
    audit_logging: bool = True
    log_successful_auth: bool = False  # Only log failures by default
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Initialize default exempt paths if not provided."""
        if self.exempt_paths is None:
            self.exempt_paths = [
                "/",
                "/health",
                "/docs",
                "/openapi.json",
                "/redoc",
                "/api/docs",
                "/api/health",
            ]


class Settings(BaseSettings):
    """Application settings with secure defaults and validation."""

    # Environment settings
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")
    testing: bool = Field(default=False, alias="TESTING")

    # Security settings
    secret_key: str = Field(default="dev-secret-key", alias="SECRET_KEY")
    enable_auth: bool = Field(default=True, alias="ENABLE_AUTH")

    # Server settings
    host: str = Field(
        default="127.0.0.1", alias="HOST"
    )  # Changed from 0.0.0.0 to fix S104
    port: int = Field(default=8080, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"], alias="CORS_ORIGINS"
    )

    # External service flags
    skip_external_services: bool = Field(default=False, alias="SKIP_EXTERNAL_SERVICES")

    # Startup configuration
    startup_timeout: int = Field(default=30, alias="STARTUP_TIMEOUT")

    # Application settings
    app_name: str = "CLARITY Digital Twin Platform"
    app_version: str = "1.0.0"

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

    # Middleware configuration
    middleware_config: MiddlewareConfig = Field(default_factory=MiddlewareConfig)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    @model_validator(mode="after")
    def validate_environment_requirements(self) -> Self:
        """Validate environment-specific requirements and set development defaults."""
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
                # Fixed G004: Using % formatting instead of f-strings in logging
                logger.warning(
                    "⚠️ Development mode: Missing credentials %s. "
                    "Using mock services (skip_external_services=%s)",
                    missing_creds,
                    self.skip_external_services,
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
                raise ValueError(msg)

        return self

    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"

    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment.lower() == "testing" or self.testing

    def should_use_mock_services(self) -> bool:
        """Check if mock services should be used instead of external services."""
        return self.skip_external_services or self.is_development()

    def get_startup_timeout(self) -> float:
        """Get the startup timeout in seconds."""
        return float(self.startup_timeout)

    def log_configuration_summary(self) -> None:
        """Log configuration summary for debugging."""
        logger.info("🔧 CLARITY Configuration Summary:")
        # Fixed G004: Using % formatting instead of f-strings in logging
        logger.info("   • Environment: %s", self.environment)
        logger.info("   • Debug mode: %s", self.debug)
        logger.info("   • Auth enabled: %s", self.enable_auth)
        logger.info("   • Skip external services: %s", self.skip_external_services)
        logger.info("   • Startup timeout: %ss", self.startup_timeout)
        logger.info("   • Firebase project: %s", self.firebase_project_id or "Not set")
        logger.info("   • GCP project: %s", self.gcp_project_id or "Not set")


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    settings = Settings()

    # Log configuration summary in debug mode
    if settings.debug or settings.log_level.upper() == "DEBUG":
        settings.log_configuration_summary()

    return settings
