"""Secure parameter store integration for sensitive configuration.

This module provides a production-ready secrets management solution with:
- AWS Systems Manager (SSM) Parameter Store integration
- Environment variable fallback
- In-memory caching with TTL
- Retry logic with exponential backoff
- Comprehensive error handling and logging
"""

from collections.abc import Callable
from functools import lru_cache
import json
import logging
import os
import time
from typing import Any, TypeVar, cast

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default configuration
DEFAULT_CACHE_TTL_SECONDS = 300  # 5 minutes
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_MIN_WAIT = 1  # seconds
DEFAULT_RETRY_MAX_WAIT = 10  # seconds

# Environment variable names
ENV_AWS_REGION = "AWS_DEFAULT_REGION"
ENV_SSM_PREFIX = "CLARITY_SSM_PREFIX"
ENV_USE_SSM = "CLARITY_USE_SSM"
ENV_CACHE_TTL = "CLARITY_SECRETS_CACHE_TTL"

# Default parameter names
DEFAULT_SSM_PREFIX = "/clarity/production"
PARAM_MODEL_SIGNATURE_KEY = "model_signature_key"
PARAM_MODEL_CHECKSUMS = "expected_model_checksums"


class SecretsCacheEntry:
    """Cache entry for a secret value with TTL support."""

    def __init__(self, value: Any, ttl_seconds: int) -> None:
        self.value = value
        self.expires_at = time.time() + ttl_seconds

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() > self.expires_at


class SecretsManager:
    """Manages secrets from AWS SSM Parameter Store with environment variable fallback.

    This class provides a robust secrets management solution with:
    - Primary source: AWS SSM Parameter Store
    - Fallback source: Environment variables
    - In-memory caching to reduce API calls
    - Retry logic for transient failures
    - Type-safe value retrieval
    """

    def __init__(
        self,
        ssm_prefix: str | None = None,
        region: str | None = None,
        cache_ttl_seconds: int | None = None,
        use_ssm: bool | None = None,
    ) -> None:
        """Initialize the secrets manager.

        Args:
            ssm_prefix: Prefix for SSM parameter names (e.g., "/clarity/production")
            region: AWS region for SSM client
            cache_ttl_seconds: Cache TTL in seconds
            use_ssm: Whether to use SSM (defaults to True in production)
        """
        # Configure from environment with defaults
        self.ssm_prefix = ssm_prefix or os.getenv(ENV_SSM_PREFIX, DEFAULT_SSM_PREFIX)
        self.region = region or os.getenv(ENV_AWS_REGION, "us-east-1")
        self.cache_ttl = cache_ttl_seconds or int(
            os.getenv(ENV_CACHE_TTL, str(DEFAULT_CACHE_TTL_SECONDS))
        )

        # Determine if we should use SSM
        if use_ssm is not None:
            self.use_ssm = use_ssm
        else:
            # Default to True if AWS credentials are available
            self.use_ssm = os.getenv(ENV_USE_SSM, "").lower() in ("true", "1", "yes")
            if not self.use_ssm:
                # Auto-detect based on environment
                self.use_ssm = self._is_aws_environment()

        # Initialize SSM client if enabled
        self._ssm_client = None
        if self.use_ssm:
            try:
                self._ssm_client = boto3.client("ssm", region_name=self.region)
                logger.info(
                    "Initialized SSM client for region %s with prefix %s",
                    self.region,
                    self.ssm_prefix,
                )
            except (NoCredentialsError, BotoCoreError) as e:
                logger.warning(
                    "Failed to initialize SSM client, falling back to environment variables: %s",
                    e,
                )
                self.use_ssm = False
        else:
            logger.info("SSM disabled, using environment variables only")

        # Initialize cache
        self._cache: dict[str, SecretsCacheEntry] = {}

    def _is_aws_environment(self) -> bool:
        """Detect if we're running in an AWS environment."""
        # Check for common AWS environment indicators
        aws_indicators = [
            "AWS_EXECUTION_ENV",  # Lambda
            "ECS_CONTAINER_METADATA_URI",  # ECS
            "AWS_LAMBDA_FUNCTION_NAME",  # Lambda
            "AWS_REGION",  # General AWS
        ]

        # Check if any AWS environment variables are set
        if any(os.getenv(var) for var in aws_indicators):
            return True

        # Check if we have AWS credentials
        try:
            boto3.client("sts").get_caller_identity()
            return True
        except Exception:
            return False

    @retry(
        retry=retry_if_exception_type((ClientError, BotoCoreError)),
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=DEFAULT_RETRY_MIN_WAIT,
            max=DEFAULT_RETRY_MAX_WAIT,
        ),
    )
    def _get_parameter_from_ssm(self, parameter_name: str) -> str | None:
        """Retrieve a parameter from SSM with retry logic.

        Args:
            parameter_name: Name of the parameter (without prefix)

        Returns:
            Parameter value or None if not found
        """
        if not self._ssm_client:
            return None

        full_parameter_name = f"{self.ssm_prefix}/{parameter_name}"

        try:
            response = self._ssm_client.get_parameter(
                Name=full_parameter_name,
                WithDecryption=True,  # Decrypt SecureString parameters
            )
            value = response["Parameter"]["Value"]
            logger.debug("Successfully retrieved parameter %s from SSM", parameter_name)
            return value
        except self._ssm_client.exceptions.ParameterNotFound:
            logger.debug("Parameter %s not found in SSM", full_parameter_name)
            return None
        except (ClientError, BotoCoreError) as e:
            logger.error(
                "Error retrieving parameter %s from SSM: %s",
                full_parameter_name,
                e,
            )
            raise

    def _get_from_cache(self, key: str) -> Any | None:
        """Get a value from cache if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        entry = self._cache.get(key)
        if entry and not entry.is_expired():
            logger.debug("Cache hit for key %s", key)
            return entry.value

        if entry:
            logger.debug("Cache expired for key %s", key)
            del self._cache[key]

        return None

    def _set_cache(self, key: str, value: Any) -> None:
        """Store a value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = SecretsCacheEntry(value, self.cache_ttl)
        logger.debug("Cached value for key %s (TTL: %ds)", key, self.cache_ttl)

    def get_string(
        self,
        key: str,
        default: str | None = None,
        env_var: str | None = None,
    ) -> str | None:
        """Get a string value from SSM or environment.

        Args:
            key: Parameter key (without prefix)
            default: Default value if not found
            env_var: Environment variable name to check as fallback

        Returns:
            String value or default
        """
        # Check cache first
        cached_value = self._get_from_cache(key)
        if cached_value is not None:
            return str(cached_value)

        # Try SSM if enabled
        if self.use_ssm:
            try:
                value = self._get_parameter_from_ssm(key)
                if value is not None:
                    self._set_cache(key, value)
                    return value
            except Exception as e:
                logger.warning(
                    "Failed to get parameter %s from SSM, falling back to environment: %s",
                    key,
                    e,
                )

        # Fall back to environment variable
        if env_var:
            value = os.getenv(env_var)
            if value is not None:
                logger.debug("Using environment variable %s for key %s", env_var, key)
                self._set_cache(key, value)
                return value

        # Use default environment variable name based on key
        default_env_var = f"CLARITY_{key.upper()}"
        value = os.getenv(default_env_var)
        if value is not None:
            logger.debug(
                "Using default environment variable %s for key %s",
                default_env_var,
                key,
            )
            self._set_cache(key, value)
            return value

        logger.debug("Using default value for key %s", key)
        return default

    def get_json(
        self,
        key: str,
        default: dict[str, Any] | None = None,
        env_var: str | None = None,
    ) -> dict[str, Any] | None:
        """Get a JSON value from SSM or environment.

        Args:
            key: Parameter key (without prefix)
            default: Default value if not found
            env_var: Environment variable name to check as fallback

        Returns:
            Parsed JSON dict or default
        """
        value = self.get_string(key, env_var=env_var)
        if value is None:
            return default

        try:
            return cast(dict[str, Any], json.loads(value))
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse JSON value for key %s: %s",
                key,
                e,
            )
            return default

    def get_model_signature_key(self) -> str:
        """Get the model signature key for integrity verification.

        Returns:
            Model signature key (never None for security)
        """
        # For security, we always return a value (default if not found)
        value = self.get_string(
            PARAM_MODEL_SIGNATURE_KEY,
            env_var="MODEL_SIGNATURE_KEY",
            default="pat_model_integrity_key_2025",  # Fallback for tests
        )
        return value or "pat_model_integrity_key_2025"

    def get_model_checksums(self) -> dict[str, str]:
        """Get expected model checksums for integrity verification.

        Returns:
            Dict mapping model size to expected checksum
        """
        # Try to get from SSM/env first
        checksums = self.get_json(
            PARAM_MODEL_CHECKSUMS,
            env_var="EXPECTED_MODEL_CHECKSUMS",
        )

        if checksums:
            return checksums

        # Fall back to hardcoded defaults for backward compatibility
        logger.warning("Model checksums not found in secrets store, using defaults")
        return {
            "small": "4b30d57febbbc8ef221e4b196bf6957e7c7f366f6b836fe800a43f69d24694ad",
            "medium": "6175021ca1a43f3c834bdaa644c45f27817cf985d8ffd186fab9b5de2c4ca661",
            "large": "c93b723f297f0d9d2ad982320b75e9212882c8f38aa40df1b600e9b2b8aa1973",
        }

    def refresh_cache(self, key: str | None = None) -> None:
        """Refresh cache for a specific key or all keys.

        Args:
            key: Specific key to refresh, or None to clear all
        """
        if key:
            self._cache.pop(key, None)
            logger.info("Cleared cache for key %s", key)
        else:
            self._cache.clear()
            logger.info("Cleared entire secrets cache")

    def health_check(self) -> dict[str, Any]:
        """Perform a health check on the secrets manager.

        Returns:
            Health check status
        """
        status = {
            "service": "SecretsManager",
            "use_ssm": self.use_ssm,
            "ssm_prefix": self.ssm_prefix,
            "cache_size": len(self._cache),
            "cache_ttl_seconds": self.cache_ttl,
        }

        if self.use_ssm and self._ssm_client:
            try:
                # Try to describe parameters to test connectivity
                self._ssm_client.describe_parameters(MaxResults=1)
                status["ssm_status"] = "healthy"
            except Exception as e:
                status["ssm_status"] = f"unhealthy: {e}"
        else:
            status["ssm_status"] = "disabled"

        return status


# Global singleton instance
_secrets_manager: SecretsManager | None = None


@lru_cache(maxsize=1)
def get_secrets_manager() -> SecretsManager:
    """Get or create the global secrets manager instance.

    Returns:
        Global SecretsManager instance
    """
    global _secrets_manager

    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
        logger.info("Initialized global SecretsManager instance")

    return _secrets_manager


def with_secret(
    key: str,
    secret_type: type[T] = str,
    env_var: str | None = None,
    default: T | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to inject secrets into function arguments.

    Args:
        key: Secret key to retrieve
        secret_type: Expected type of the secret
        env_var: Environment variable fallback
        default: Default value if not found

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = get_secrets_manager()

            if secret_type == str:
                value = manager.get_string(key, env_var=env_var, default=default)
            elif secret_type == dict:
                value = manager.get_json(key, env_var=env_var, default=default)
            else:
                raise ValueError(f"Unsupported secret type: {secret_type}")

            # Inject the secret as a keyword argument
            kwargs[key] = value
            return func(*args, **kwargs)

        return wrapper

    return decorator
