"""CLARITY Digital Twin Platform - AWS Configuration Management.

Environment-based configuration using Pydantic settings for AWS deployment.
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
    """Configuration for AWS Cognito authentication middleware."""

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
    """Application settings for AWS deployment."""

    # Environment settings
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")
    testing: bool = Field(default=False, alias="TESTING")

    # Security settings
    secret_key: str = Field(default="dev-secret-key", alias="SECRET_KEY")
    enable_auth: bool = Field(default=True, alias="ENABLE_AUTH")
    enable_self_signup: bool = Field(default=False, alias="ENABLE_SELF_SIGNUP")

    # Server settings
    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=8080, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"],
        alias="CORS_ORIGINS",
    )

    # External service flags
    skip_external_services: bool = Field(default=False, alias="SKIP_EXTERNAL_SERVICES")

    # Startup configuration
    startup_timeout: int = Field(default=30, alias="STARTUP_TIMEOUT")

    # Application settings
    app_name: str = "CLARITY Digital Twin Platform"
    app_version: str = "1.0.0"

    # AWS Core Settings
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    aws_access_key_id: str | None = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(
        default=None, alias="AWS_SECRET_ACCESS_KEY"
    )
    aws_session_token: str | None = Field(default=None, alias="AWS_SESSION_TOKEN")

    # AWS Cognito Settings
    cognito_user_pool_id: str = Field(
        default="us-east-1_efXaR5EcP", alias="COGNITO_USER_POOL_ID"
    )
    cognito_client_id: str = Field(
        default="7sm7ckrkovg78b03n1595euc71", alias="COGNITO_CLIENT_ID"
    )
    cognito_region: str | None = Field(default="us-east-1", alias="COGNITO_REGION")

    # DynamoDB Settings
    dynamodb_table_name: str = Field(
        default="clarity-health-data", alias="DYNAMODB_TABLE_NAME"
    )
    dynamodb_endpoint_url: str | None = Field(
        default=None,
        alias="DYNAMODB_ENDPOINT_URL",
        description="DynamoDB endpoint URL (for local testing with DynamoDB Local)",
    )

    # S3 Settings (replacing Cloud Storage)
    s3_bucket_name: str = Field(
        default="clarity-health-uploads", alias="S3_BUCKET_NAME"
    )
    s3_endpoint_url: str | None = Field(
        default=None,
        alias="S3_ENDPOINT_URL",
        description="S3 endpoint URL (for local testing with LocalStack)",
    )

    # SQS/SNS Settings (replacing Pub/Sub)
    sqs_queue_url: str | None = Field(default=None, alias="SQS_QUEUE_URL")
    sns_topic_arn: str | None = Field(default=None, alias="SNS_TOPIC_ARN")
    sqs_endpoint_url: str | None = Field(
        default=None,
        alias="SQS_ENDPOINT_URL",
        description="SQS endpoint URL (for local testing)",
    )

    # Gemini API Settings (keeping this for AI functionality)
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.7, alias="GEMINI_TEMPERATURE")
    gemini_max_tokens: int = Field(default=1000, alias="GEMINI_MAX_TOKENS")

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
        # In development, warn about missing credentials but don't fail
        if self.environment.lower() == "development":
            missing_creds: list[str] = []

            if self.enable_auth and not self.cognito_user_pool_id:
                missing_creds.append("COGNITO_USER_POOL_ID")
            if self.enable_auth and not self.cognito_client_id:
                missing_creds.append("COGNITO_CLIENT_ID")
            if not self.skip_external_services and not self.dynamodb_table_name:
                missing_creds.append("DYNAMODB_TABLE_NAME")
            if not self.skip_external_services and not self.s3_bucket_name:
                missing_creds.append("S3_BUCKET_NAME")
            if not self.gemini_api_key:
                missing_creds.append("GEMINI_API_KEY")

            if missing_creds:
                logger.warning(
                    "⚠️ Development mode: Missing credentials %s. "
                    "Using mock services (skip_external_services=%s)",
                    missing_creds,
                    self.skip_external_services,
                )

        # In production, require critical credentials
        elif self.environment.lower() == "production":
            required_for_production: list[str] = []

            if self.enable_auth and not self.cognito_user_pool_id:
                required_for_production.append("COGNITO_USER_POOL_ID (auth enabled)")
            if self.enable_auth and not self.cognito_client_id:
                required_for_production.append("COGNITO_CLIENT_ID (auth enabled)")

            if not self.skip_external_services:
                if not self.dynamodb_table_name:
                    required_for_production.append("DYNAMODB_TABLE_NAME")
                if not self.s3_bucket_name:
                    required_for_production.append("S3_BUCKET_NAME")
                if not self.gemini_api_key:
                    required_for_production.append("GEMINI_API_KEY")

            if required_for_production:
                missing_vars = ", ".join(required_for_production)
                msg = (
                    f"Production environment requires: {missing_vars}. "
                    f"Set SKIP_EXTERNAL_SERVICES=true to use mock services."
                )
                raise ValueError(msg)

        # Set Cognito region to AWS region if not specified
        if not self.cognito_region:
            self.cognito_region = self.aws_region

        # Configure middleware settings based on environment
        if self.environment.lower() == "development":
            # Development environment: permissive settings for debugging
            self.middleware_config = MiddlewareConfig(
                enabled=self.enable_auth,
                graceful_degradation=False,  # Show actual auth errors
                fallback_to_mock=True,
                log_successful_auth=True,
                cache_enabled=False,  # Disable cache for easier debugging
                initialization_timeout_seconds=10,  # Longer timeout
                exempt_paths=self.middleware_config.exempt_paths,  # Preserve exempt paths
                cache_ttl_seconds=self.middleware_config.cache_ttl_seconds,
                cache_max_size=self.middleware_config.cache_max_size,
                audit_logging=self.middleware_config.audit_logging,
                log_level=self.middleware_config.log_level,
            )
        elif self.environment.lower() == "testing":
            # Testing environment: use mock auth
            self.middleware_config = MiddlewareConfig(
                enabled=self.enable_auth,
                graceful_degradation=False,  # Show actual auth errors
                fallback_to_mock=True,
                log_successful_auth=False,
                cache_enabled=True,  # Keep default
                initialization_timeout_seconds=self.middleware_config.initialization_timeout_seconds,
                exempt_paths=self.middleware_config.exempt_paths,
                cache_ttl_seconds=self.middleware_config.cache_ttl_seconds,
                cache_max_size=self.middleware_config.cache_max_size,
                audit_logging=self.middleware_config.audit_logging,
                log_level=self.middleware_config.log_level,
            )
        elif self.environment.lower() == "production":
            # Production environment: strict settings
            self.middleware_config = MiddlewareConfig(
                enabled=self.enable_auth,
                graceful_degradation=False,  # Fail fast
                fallback_to_mock=False,  # No mock fallback
                log_successful_auth=False,  # Only log failures
                cache_enabled=True,  # Enable cache for performance
                cache_ttl_seconds=300,  # 5 minutes
                initialization_timeout_seconds=5,  # Shorter timeout
                exempt_paths=self.middleware_config.exempt_paths,
                cache_max_size=self.middleware_config.cache_max_size,
                audit_logging=self.middleware_config.audit_logging,
                log_level=self.middleware_config.log_level,
            )
        else:
            # For other environments, just update enabled status
            self.middleware_config.enabled = self.enable_auth

        return self

    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"

    def is_testing(self) -> bool:
        """Check if running in test mode."""
        return self.testing or self.environment.lower() == "testing"

    def should_use_mock_services(self) -> bool:
        """Determine if mock services should be used."""
        return self.skip_external_services or self.is_testing()

    def get_cors_origins(self) -> list[str]:
        """Get configured CORS origins."""
        if self.is_production():
            # In production, use specific origins
            return self.cors_origins
        # In development, allow common local origins
        return [
            "http://localhost:3000",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
        ]


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Module-level settings instance
settings = get_settings()
