"""Adapter to convert between ClarityConfig and Settings."""

import os

from clarity.core.config_aws import Settings
from clarity.startup.config_schema import ClarityConfig


def clarity_config_to_settings(config: ClarityConfig) -> Settings:
    """Convert ClarityConfig to Settings for DI container initialization.

    Args:
        config: ClarityConfig instance from startup

    Returns:
        Settings instance compatible with AWS container
    """
    # Temporarily set environment variables from ClarityConfig
    # This allows Settings to pick them up via Pydantic BaseSettings
    env_overrides = {
        "ENVIRONMENT": config.environment,
        "DEBUG": str(config.debug),
        "TESTING": str(config.is_testing()),
        "SECRET_KEY": config.security.secret_key,
        "ENABLE_AUTH": str(config.enable_auth),
        "LOG_LEVEL": config.log_level,
        "CORS_ORIGINS": '["' + '","'.join(config.security.cors_origins) + '"]',
        "SKIP_EXTERNAL_SERVICES": str(config.skip_external_services),
        "STARTUP_TIMEOUT": str(config.startup_timeout),
        "AWS_REGION": config.aws.region,
        "COGNITO_USER_POOL_ID": config.cognito.user_pool_id,
        "COGNITO_CLIENT_ID": config.cognito.client_id,
        "COGNITO_REGION": config.cognito.region or config.aws.region,
        "DYNAMODB_TABLE_NAME": config.dynamodb.table_name,
        "S3_BUCKET_NAME": config.s3.bucket_name,
        "GEMINI_API_KEY": config.gemini.api_key or "",
        "GEMINI_MODEL": config.gemini.model,
        "GEMINI_TEMPERATURE": str(config.gemini.temperature),
        "GEMINI_MAX_TOKENS": str(config.gemini.max_tokens),
    }

    # Save current environment
    original_env = {k: os.environ.get(k) for k in env_overrides}

    try:
        # Set environment variables
        for key, value in env_overrides.items():
            if value is not None:
                os.environ[key] = value

        # Create Settings instance (will read from environment)
        return Settings()
    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
