"""CLARITY Configuration Schema.

Comprehensive Pydantic-based configuration validation with clear error messages.
Validates all environment variables and service configurations before startup.
"""

from __future__ import annotations

from enum import StrEnum
import logging
import os
from typing import Any, ClassVar
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Constants
AWS_REGION_MIN_PARTS = 3
COGNITO_CLIENT_ID_MIN_LENGTH = 20


class Environment(StrEnum):
    """Valid environment values."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    """Valid log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AWSConfig(BaseModel):
    """AWS service configuration with validation."""

    region: str = Field(
        default="us-east-1",
        description="AWS region for all services",
        min_length=1,
    )
    access_key_id: str = Field(
        default="",
        description="AWS access key ID (optional with IAM roles)",
    )
    secret_access_key: str = Field(
        default="",
        description="AWS secret access key (optional with IAM roles)",
        repr=False,  # Don't include in string representation
    )
    session_token: str = Field(
        default="",
        description="AWS session token for temporary credentials",
        repr=False,
    )

    @field_validator("region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        """Validate AWS region format."""
        if not v:
            msg = "AWS region cannot be empty"
            raise ValueError(msg)
        # Basic AWS region format validation
        if len(v.split("-")) < AWS_REGION_MIN_PARTS:
            msg = f"Invalid AWS region format: {v}"
            raise ValueError(msg)
        return v


class CognitoConfig(BaseModel):
    """AWS Cognito configuration with validation."""

    user_pool_id: str = Field(
        default="",
        description="Cognito User Pool ID",
        min_length=0,
    )
    client_id: str = Field(
        default="",
        description="Cognito App Client ID",
        min_length=0,
    )
    region: str = Field(
        default="",
        description="Cognito region (defaults to AWS region)",
        min_length=0,
    )

    @field_validator("user_pool_id")
    @classmethod
    def validate_user_pool_id(cls, v: str) -> str:
        """Validate Cognito User Pool ID format."""
        if v and not v.startswith(("us-", "eu-", "ap-", "ca-", "sa-")):
            msg = f"Invalid Cognito User Pool ID format: {v}"
            raise ValueError(msg)
        return v

    @field_validator("client_id")
    @classmethod
    def validate_client_id(cls, v: str) -> str:
        """Validate Cognito Client ID format."""
        if v and len(v) < COGNITO_CLIENT_ID_MIN_LENGTH:
            msg = f"Cognito Client ID appears invalid: {v}"
            raise ValueError(msg)
        return v


class DynamoDBConfig(BaseModel):
    """DynamoDB configuration with validation."""

    table_name: str = Field(
        default="clarity-health-data",
        description="DynamoDB table name",
        min_length=1,
    )
    endpoint_url: str = Field(
        default="",
        description="DynamoDB endpoint URL (for local testing)",
    )

    @field_validator("table_name")
    @classmethod
    def validate_table_name(cls, v: str) -> str:
        """Validate DynamoDB table name."""
        if not v:
            msg = "DynamoDB table name cannot be empty"
            raise ValueError(msg)
        # Basic table name validation
        if not v.replace("-", "").replace("_", "").isalnum():
            msg = f"Invalid DynamoDB table name: {v}"
            raise ValueError(msg)
        return v

    @field_validator("endpoint_url")
    @classmethod
    def validate_endpoint_url(cls, v: str) -> str:
        """Validate DynamoDB endpoint URL."""
        if v:
            parsed = urlparse(v)
            if not parsed.scheme or not parsed.netloc:
                msg = f"Invalid DynamoDB endpoint URL: {v}"
                raise ValueError(msg)
        return v


class S3Config(BaseModel):
    """S3 configuration with validation."""

    bucket_name: str = Field(
        default="clarity-health-uploads",
        description="S3 bucket name",
        min_length=1,
    )
    ml_models_bucket: str = Field(
        default="clarity-ml-models-124355672559",
        description="S3 bucket for ML models",
        min_length=1,
    )
    endpoint_url: str = Field(
        default="",
        description="S3 endpoint URL (for local testing)",
    )

    @field_validator("bucket_name", "ml_models_bucket")
    @classmethod
    def validate_bucket_name(cls, v: str) -> str:
        """Validate S3 bucket name."""
        if not v:
            msg = "S3 bucket name cannot be empty"
            raise ValueError(msg)
        # Basic S3 bucket name validation
        if not v.replace("-", "").replace(".", "").isalnum():
            msg = f"Invalid S3 bucket name: {v}"
            raise ValueError(msg)
        return v


class GeminiConfig(BaseModel):
    """Gemini AI configuration with validation."""

    api_key: str = Field(
        default="",
        description="Gemini API key",
        repr=False,
    )
    model: str = Field(
        default="gemini-1.5-flash",
        description="Gemini model name",
    )
    temperature: float = Field(
        default=0.7,
        description="Gemini temperature",
        ge=0.0,
        le=2.0,
    )
    max_tokens: int = Field(
        default=1000,
        description="Maximum tokens for Gemini responses",
        gt=0,
        le=32768,
    )


class SecurityConfig(BaseModel):
    """Security configuration with validation."""

    secret_key: str = Field(
        default="dev-secret-key",
        description="Application secret key",
        min_length=8,
        repr=False,
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"],
        description="CORS allowed origins",
    )

    @model_validator(mode="before")
    @classmethod
    def parse_cors_origins(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Parse CORS origins from comma-separated string."""
        if "cors_origins" in values and isinstance(values["cors_origins"], str):
            values["cors_origins"] = [
                o.strip() for o in values["cors_origins"].split(",") if o.strip()
            ]
        return values

    max_request_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum request size in bytes",
        gt=0,
    )
    rate_limit_requests: int = Field(
        default=100,
        description="Rate limit requests per minute",
        gt=0,
    )

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, v: list[str]) -> list[str]:
        """Validate CORS origins."""
        for origin in v:
            if "*" in origin and origin != "*":
                msg = f"Invalid CORS origin with partial wildcard: {origin}"
                raise ValueError(msg)
            if origin != "*":
                parsed = urlparse(origin)
                if not parsed.scheme or not parsed.netloc:
                    msg = f"Invalid CORS origin URL: {origin}"
                    raise ValueError(msg)
        return v

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate secret key strength."""
        if (
            v == "dev-secret-key"
            and os.getenv("ENVIRONMENT", "").lower() == "production"
        ):
            msg = "Must set custom SECRET_KEY in production"
            raise ValueError(msg)
        return v


class ClarityConfig(BaseSettings):
    """Comprehensive CLARITY configuration with bulletproof validation.

    This configuration schema validates all environment variables and service
    configurations before startup, providing clear error messages for any issues.
    """

    # Environment and basic settings
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment",
        alias="ENVIRONMENT",
    )
    debug: bool = Field(default=False, description="Enable debug mode", alias="DEBUG")
    testing: bool = Field(
        default=False, description="Enable testing mode", alias="TESTING"
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO, description="Logging level", alias="LOG_LEVEL"
    )

    # Server settings
    host: str = Field(default="127.0.0.1", description="Server host", alias="HOST")
    port: int = Field(
        default=8000, description="Server port", ge=1024, le=65535, alias="PORT"
    )

    # Feature flags
    enable_auth: bool = Field(
        default=True, description="Enable authentication", alias="ENABLE_AUTH"
    )
    skip_external_services: bool = Field(
        default=False,
        description="Skip external service initialization (use mocks)",
        alias="SKIP_EXTERNAL_SERVICES",
    )
    skip_aws_init: bool = Field(
        default=False,
        description="Skip AWS service initialization",
        alias="SKIP_AWS_INIT",
    )

    # Service configurations
    aws: AWSConfig = Field(default_factory=AWSConfig, description="AWS configuration")
    cognito: CognitoConfig = Field(
        default_factory=CognitoConfig, description="Cognito configuration"
    )
    dynamodb: DynamoDBConfig = Field(
        default_factory=DynamoDBConfig, description="DynamoDB configuration"
    )
    s3: S3Config = Field(default_factory=S3Config, description="S3 configuration")
    gemini: GeminiConfig = Field(
        default_factory=GeminiConfig, description="Gemini AI configuration"
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig, description="Security configuration"
    )

    # Startup settings
    startup_timeout: int = Field(
        default=30,
        description="Startup timeout in seconds",
        gt=0,
        le=300,
        alias="STARTUP_TIMEOUT",
    )
    health_check_timeout: int = Field(
        default=5,
        description="Health check timeout per service",
        gt=0,
        le=30,
        alias="HEALTH_CHECK_TIMEOUT",
    )

    # Validation error tracking
    _validation_errors: ClassVar[list[str]] = []

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",  # Allow AWS__REGION format
        extra="allow",
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def extract_nested_config(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Extract nested configuration from environment variables."""
        # Debug: log what values we receive
        import os  # noqa: PLC0415 - Debug logging

        # For Pydantic BaseSettings, we need to check both the values dict and env vars
        aws_config = {}
        cognito_config = {}
        dynamodb_config = {}
        s3_config = {}
        gemini_config = {}
        security_config = {}

        # Extract AWS config - check env vars directly since BaseSettings might not pass them
        aws_config["region"] = values.get("AWS_REGION") or os.getenv(
            "AWS_REGION", "us-east-1"
        )
        aws_config["access_key_id"] = values.get("AWS_ACCESS_KEY_ID", "")
        aws_config["secret_access_key"] = values.get("AWS_SECRET_ACCESS_KEY", "")
        aws_config["session_token"] = values.get("AWS_SESSION_TOKEN", "")

        # Extract Cognito config - check both values and env vars
        cognito_config["user_pool_id"] = values.get(
            "COGNITO_USER_POOL_ID"
        ) or os.getenv("COGNITO_USER_POOL_ID", "")
        cognito_config["client_id"] = values.get("COGNITO_CLIENT_ID") or os.getenv(
            "COGNITO_CLIENT_ID", ""
        )
        cognito_config["region"] = values.get("COGNITO_REGION") or os.getenv(
            "COGNITO_REGION", aws_config["region"]
        )

        # Extract DynamoDB config - check both values and env vars
        dynamodb_config["table_name"] = values.get("DYNAMODB_TABLE_NAME") or os.getenv(
            "DYNAMODB_TABLE_NAME", "clarity-health-data"
        )
        dynamodb_config["endpoint_url"] = values.get(
            "DYNAMODB_ENDPOINT_URL"
        ) or os.getenv("DYNAMODB_ENDPOINT_URL", "")

        # Extract S3 config - check both values and env vars
        s3_config["bucket_name"] = values.get("S3_BUCKET_NAME") or os.getenv(
            "S3_BUCKET_NAME", "clarity-health-uploads"
        )
        s3_config["ml_models_bucket"] = values.get("S3_ML_MODELS_BUCKET") or os.getenv(
            "S3_ML_MODELS_BUCKET", "clarity-ml-models-124355672559"
        )
        s3_config["endpoint_url"] = values.get("S3_ENDPOINT_URL") or os.getenv(
            "S3_ENDPOINT_URL", ""
        )

        # Extract Gemini config - check both values and env vars
        gemini_config["api_key"] = values.get("GEMINI_API_KEY") or os.getenv(
            "GEMINI_API_KEY", ""
        )
        gemini_config["model"] = values.get("GEMINI_MODEL") or os.getenv(
            "GEMINI_MODEL", "gemini-1.5-flash"
        )
        temp_val = values.get("GEMINI_TEMPERATURE") or os.getenv(
            "GEMINI_TEMPERATURE", "0.7"
        )
        gemini_config["temperature"] = float(str(temp_val))

        tokens_val = values.get("GEMINI_MAX_TOKENS") or os.getenv(
            "GEMINI_MAX_TOKENS", "1000"
        )
        gemini_config["max_tokens"] = int(str(tokens_val))

        # Extract security config - check both values and env vars
        security_config["secret_key"] = values.get("SECRET_KEY") or os.getenv(
            "SECRET_KEY", "dev-secret-key"
        )
        cors_origins_str = values.get("CORS_ALLOWED_ORIGINS") or os.getenv(
            "CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080"
        )
        # Pass raw string - SecurityConfig will parse and validate it
        security_config["cors_origins"] = cors_origins_str
        security_config["max_request_size"] = int(
            values.get("MAX_REQUEST_SIZE", str(10 * 1024 * 1024))
        )
        security_config["rate_limit_requests"] = int(
            values.get("RATE_LIMIT_REQUESTS", "100")
        )

        # Update values with nested configs
        values["aws"] = aws_config
        values["cognito"] = cognito_config
        values["dynamodb"] = dynamodb_config
        values["s3"] = s3_config
        values["gemini"] = gemini_config
        values["security"] = security_config

        return values

    @model_validator(mode="after")
    def validate_environment_requirements(self) -> "ClarityConfig":
        """Validate environment-specific requirements."""
        env = self.environment
        skip_external = self.skip_external_services
        enable_auth = self.enable_auth

        validation_errors = []

        # Production requirements
        if env == Environment.PRODUCTION and not skip_external:
            if enable_auth:
                if not self.cognito.user_pool_id:
                    validation_errors.append(
                        "COGNITO_USER_POOL_ID required in production with auth enabled"
                    )
                if not self.cognito.client_id:
                    validation_errors.append(
                        "COGNITO_CLIENT_ID required in production with auth enabled"
                    )

            # Check AWS region
            if not self.aws.region:
                validation_errors.append("AWS_REGION required in production")

            # Check required buckets
            if not self.s3.bucket_name:
                validation_errors.append("S3_BUCKET_NAME required in production")
            if not self.s3.ml_models_bucket:
                validation_errors.append("S3_ML_MODELS_BUCKET required in production")

            # Check secret key
            if self.security.secret_key == "dev-secret-key":  # noqa: S105
                validation_errors.append("Custom SECRET_KEY required in production")

        # Store validation errors for reporting
        type(self)._validation_errors = validation_errors  # noqa: SLF001

        if validation_errors:
            error_msg = "\n".join([f"  • {error}" for error in validation_errors])
            msg = f"Configuration validation failed:\n{error_msg}"
            raise ValueError(msg)

        return self

    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == Environment.DEVELOPMENT

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == Environment.PRODUCTION

    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment == Environment.TESTING or self.testing

    def should_use_mock_services(self) -> bool:
        """Determine if mock services should be used."""
        return self.skip_external_services or self.is_development() or self.is_testing()

    def get_service_requirements(self) -> dict[str, bool]:
        """Get service requirements based on configuration."""
        return {
            "cognito": self.enable_auth and not self.should_use_mock_services(),
            "dynamodb": not self.should_use_mock_services(),
            "s3": not self.should_use_mock_services(),
            "gemini": bool(self.gemini.api_key),
        }

    def get_startup_summary(self) -> dict[str, Any]:
        """Get startup configuration summary."""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "auth_enabled": self.enable_auth,
            "mock_services": self.should_use_mock_services(),
            "startup_timeout": self.startup_timeout,
            "required_services": self.get_service_requirements(),
            "aws_region": self.aws.region,
            "log_level": self.log_level.value,
        }

    @classmethod
    def validate_from_env(cls) -> tuple[ClarityConfig | None, list[str]]:
        """Validate configuration from environment variables.

        Returns:
            Tuple of (config, errors). Config is None if validation fails.
        """
        try:
            # Use Pydantic's proper environment loading by creating instance with no args
            # This will read from environment variables using the aliases defined in fields
            config = cls()
            return config, []
        except ValidationError as e:
            errors = []
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])
                errors.append(f"{field_path}: {error['msg']}")
            return None, errors
        except (ValueError, TypeError, RuntimeError) as e:
            errors = getattr(cls, "_validation_errors", [])
            if not errors:
                errors = [str(e)]
            return None, errors


def load_config() -> ClarityConfig:
    """Load and validate configuration with clear error reporting."""
    config, errors = ClarityConfig.validate_from_env()

    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error("  • %s", error)
        msg = "Configuration validation failed - see logs for details"
        raise ValueError(msg)

    if config is None:
        msg = "Failed to load configuration"
        raise ValueError(msg)

    logger.info("Configuration loaded successfully")
    return config
