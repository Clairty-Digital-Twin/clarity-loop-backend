"""CLARITY Digital Twin Platform - Configuration Management.

Environment-based configuration using Pydantic settings for secure,
production-ready deployment across development, staging, and production.
"""

# removed - breaks FastAPI

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
    """Configuration for AWS Cognito authentication middleware.

    Contains all middleware-specific settings for authentication,
    token caching, and error handling.
    """

    # Authentication settings
    enabled: bool = True

    # Exempt paths (no authentication required)
    exempt_paths: list[str] | None = None

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
    enable_self_signup: bool = Field(default=False, alias="ENABLE_SELF_SIGNUP")

    # Server settings
    host: str = Field(
        default="127.0.0.1", alias="HOST"
    )  # Changed from 0.0.0.0 to fix S104
    port: int = Field(default=8080, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # CORS Security Settings - HARDENED CONFIGURATION
    cors_allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        alias="CORS_ALLOWED_ORIGINS",
        description="Explicitly allowed origins for CORS - no wildcards for security",
    )
    cors_allow_credentials: bool = Field(
        default=True,
        alias="CORS_ALLOW_CREDENTIALS",
        description="Allow credentials in CORS requests (secure with explicit origins)",
    )
    cors_allowed_methods: list[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        alias="CORS_ALLOWED_METHODS",
        description="Explicitly allowed HTTP methods - no wildcards",
    )
    cors_allowed_headers: list[str] = Field(
        default_factory=lambda: [
            "Authorization",
            "Content-Type",
            "Accept",
            "X-Requested-With",
        ],
        alias="CORS_ALLOWED_HEADERS",
        description="Explicitly allowed headers - no wildcards",
    )
    cors_max_age: int = Field(
        default=86400,
        alias="CORS_MAX_AGE",
        description="Cache preflight requests for 24 hours (86400 seconds)",
    )

    # Request Size Limits - DoS Protection
    max_request_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        alias="MAX_REQUEST_SIZE",
        description="Maximum request body size in bytes (default 10MB)",
    )
    max_json_size: int = Field(
        default=5 * 1024 * 1024,  # 5MB
        alias="MAX_JSON_SIZE",
        description="Maximum JSON payload size in bytes (default 5MB)",
    )
    max_upload_size: int = Field(
        default=50 * 1024 * 1024,  # 50MB
        alias="MAX_UPLOAD_SIZE",
        description="Maximum file upload size in bytes (default 50MB)",
    )
    max_form_size: int = Field(
        default=1024 * 1024,  # 1MB
        alias="MAX_FORM_SIZE",
        description="Maximum form data size in bytes (default 1MB)",
    )

    # External service flags
    skip_external_services: bool = Field(default=False, alias="SKIP_EXTERNAL_SERVICES")

    # Startup configuration
    startup_timeout: int = Field(default=30, alias="STARTUP_TIMEOUT")

    # Application settings
    app_name: str = "CLARITY Digital Twin Platform"
    app_version: str = "1.0.0"

    # AWS settings
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    aws_access_key_id: str = Field(default="", alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(default="", alias="AWS_SECRET_ACCESS_KEY")

    # AWS Cognito settings
    cognito_user_pool_id: str = Field(default="", alias="COGNITO_USER_POOL_ID")
    cognito_client_id: str = Field(default="", alias="COGNITO_CLIENT_ID")
    cognito_region: str = Field(default="us-east-1", alias="COGNITO_REGION")

    # AWS DynamoDB settings
    dynamodb_table_name: str = Field(
        default="clarity-health-data", alias="DYNAMODB_TABLE_NAME"
    )
    dynamodb_region: str = Field(default="us-east-1", alias="DYNAMODB_REGION")

    # AWS S3 settings
    s3_bucket_name: str = Field(
        default="clarity-health-uploads", alias="S3_BUCKET_NAME"
    )
    s3_region: str = Field(default="us-east-1", alias="S3_REGION")

    # Google Cloud settings (for Gemini AI only)
    # Removed GCP project ID - now using AWS
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
        "extra": "allow",  # Allow extra inputs for testing flexibility
    }

    @model_validator(mode="after")
    def validate_environment_requirements(self) -> Self:
        """Validate environment-specific requirements and set development defaults."""
        # If in testing mode, and no path is provided, set a mock path.
        if self.is_testing():
            if not self.google_application_credentials:
                self.google_application_credentials = "mock_google_creds.json"
            return self

        # In development, warn about missing credentials but don't fail
        if self.environment.lower() == "development":
            missing_creds: list[str] = []

            if self.enable_auth and not self.cognito_user_pool_id:
                missing_creds.append("COGNITO_USER_POOL_ID")
            if self.enable_auth and not self.cognito_client_id:
                missing_creds.append("COGNITO_CLIENT_ID")
            if not self.aws_region:
                missing_creds.append("AWS_REGION")

            if missing_creds:
                # Fixed G004: Using % formatting instead of f-strings in logging
                logger.warning(
                    "⚠️ Development mode: Missing AWS credentials %s. "
                    "Using mock services (skip_external_services=%s)",
                    missing_creds,
                    self.skip_external_services,
                )

        # In production, require critical credentials (unless skipping external services)
        elif self.environment.lower() == "production":
            # Skip validation if external services are disabled
            if self.skip_external_services:
                logger.warning(
                    "⚠️ Production mode with SKIP_EXTERNAL_SERVICES=true - using mock services"
                )
                return self

            required_for_production: list[str] = []

            if self.enable_auth and not self.cognito_user_pool_id:
                required_for_production.append("COGNITO_USER_POOL_ID (auth enabled)")
            if self.enable_auth and not self.cognito_client_id:
                required_for_production.append("COGNITO_CLIENT_ID (auth enabled)")
            if not self.aws_region:
                required_for_production.append("AWS_REGION")

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
        return self.environment.lower() == "testing" or (
            self.testing and self.environment.lower() != "production"
        )

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
        logger.info("   • AWS region: %s", self.aws_region or "Not set")
        logger.info(
            "   • Cognito User Pool: %s", self.cognito_user_pool_id or "Not set"
        )
        logger.info("   • DynamoDB table: %s", self.dynamodb_table_name or "Not set")

    @property
    def get_cors_origins(self) -> list[str]:
        """Parse CORS origins from comma-separated string to list."""
        # Split comma-separated string and strip whitespace
        origins = [
            origin.strip()
            for origin in self.cors_allowed_origins.split(",")
            if origin.strip()
        ]

        # Security validation: no wildcards allowed
        for origin in origins:
            if "*" in origin:
                msg = (
                    f"Security violation: Wildcard origin '{origin}' detected. "
                    f"CORS origins must be explicitly specified for security."
                )
                raise ValueError(msg)

        return origins

    def get_middleware_config(self) -> MiddlewareConfig:
        """Get middleware configuration based on environment.

        Returns middleware configuration optimized for the current environment.
        Development environments use more lenient settings for easier debugging,
        while production environments use stricter security settings.
        """
        config = MiddlewareConfig()

        if self.is_development():
            # Development environment - lenient settings for debugging
            config.enabled = self.enable_auth
            config.graceful_degradation = (
                False  # <-- DEBUGGING: Show actual auth errors
            )
            config.fallback_to_mock = True
            config.cache_enabled = False  # Disable cache for easier debugging
            config.log_successful_auth = True  # Log all auth attempts
            config.log_level = "DEBUG"
            config.initialization_timeout_seconds = 10  # Longer timeout for debugging

        elif self.is_production():
            # Production environment - strict security settings
            config.enabled = self.enable_auth
            config.graceful_degradation = (
                False  # <-- DEBUGGING: Show actual auth errors
            )
            config.fallback_to_mock = False  # Never fall back in production
            config.cache_enabled = True
            config.cache_ttl_seconds = 300  # 5 minutes
            config.cache_max_size = 1000
            config.log_successful_auth = False  # Only log failures
            config.log_level = "INFO"
            config.initialization_timeout_seconds = 5

        else:
            # Unknown environment - use conservative defaults
            config.enabled = self.enable_auth
            config.graceful_degradation = (
                False  # <-- DEBUGGING: Show actual auth errors
            )
            config.fallback_to_mock = True

        return config


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    settings = Settings()

    # Log configuration summary in debug mode
    if settings.debug or settings.log_level.upper() == "DEBUG":
        settings.log_configuration_summary()

    return settings
