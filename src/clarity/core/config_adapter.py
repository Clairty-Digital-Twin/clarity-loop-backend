"""Adapter to convert between ClarityConfig and Settings."""

from clarity.core.config_aws import Settings
from clarity.startup.config_schema import ClarityConfig


def clarity_config_to_settings(config: ClarityConfig) -> Settings:
    """Convert ClarityConfig to Settings for DI container initialization.
    
    Args:
        config: ClarityConfig instance from startup
        
    Returns:
        Settings instance compatible with AWS container
    """
    # Map ClarityConfig fields to Settings
    settings_dict = {
        # Environment settings
        "environment": config.environment,
        "debug": config.debug,
        "testing": config.is_testing,
        
        # Security settings
        "secret_key": config.security.secret_key,
        "enable_auth": config.enable_auth,
        
        # Server settings
        "host": "127.0.0.1",  # Default as not in ClarityConfig
        "port": 8080,  # Default as not in ClarityConfig
        "log_level": config.log_level,
        "cors_origins": config.security.cors_origins,
        
        # External service flags
        "skip_external_services": config.skip_external_services,
        
        # Startup configuration
        "startup_timeout": int(config.health_check.startup_timeout),
        
        # AWS settings (from ClarityConfig.aws)
        "aws_region": config.aws.region,
        "cognito_user_pool_id": config.aws.cognito_user_pool_id,
        "cognito_client_id": config.aws.cognito_client_id,
        "cognito_region": config.aws.region,
        "dynamodb_table_name": config.aws.dynamodb_table_name,
        "s3_bucket_name": config.aws.s3_bucket_name,
        
        # Gemini settings
        "gemini_api_key": config.ml.gemini_api_key,
        "gemini_model": config.ml.gemini_model,
        "gemini_temperature": config.ml.gemini_temperature,
        "gemini_max_tokens": config.ml.gemini_max_tokens,
    }
    
    # Create Settings instance
    return Settings(**settings_dict)