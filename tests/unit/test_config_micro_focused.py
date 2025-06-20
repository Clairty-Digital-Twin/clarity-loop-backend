"""MICRO-FOCUSED Config Tests - CHUNK 3.

🚀 TINY CHUNK FOR CONFIG MODULE 🚀
Target: config.py 58% → 70%

Breaking down into MICRO pieces:
- Basic config creation
- Environment variable handling
- Simple validation paths
- Error conditions

SMALLEST POSSIBLE TESTS!
"""

from __future__ import annotations

import os
from unittest.mock import patch

from clarity.core.config import MiddlewareConfig, Settings, get_settings
from clarity.core.exceptions import (
    InvalidConfigurationError,
    MissingConfigurationError,
)


class TestBasicConfigCreation:
    """Test basic config object creation - MICRO CHUNK 3A."""

    @staticmethod
    def test_config_creation_with_defaults() -> None:
        """Test creating config with default values."""
        # Act
        config = Settings()

        # Assert
        assert config is not None
        assert hasattr(config, "environment")

    @staticmethod
    def test_config_has_required_attributes() -> None:
        """Test config has all required attributes."""
        # Act
        config = Settings()

        # Assert - Check for key attributes
        assert hasattr(config, "environment")
        assert hasattr(config, "debug")
        assert hasattr(config, "log_level")
        assert hasattr(config, "port")
        assert hasattr(config, "host")
        assert hasattr(config, "skip_external_services")

    @staticmethod
    def test_config_environment_defaults() -> None:
        """Test default environment configuration."""
        # Act
        config = Settings()

        # Assert
        assert config.environment in {"development", "production", "testing"}
        assert config.log_level in {"DEBUG", "INFO", "WARNING", "ERROR"}

    @staticmethod
    def test_config_boolean_attributes() -> None:
        """Test boolean configuration attributes."""
        # Act
        config = Settings()

        # Assert boolean attributes exist and are proper booleans
        assert isinstance(config.debug, bool)
        assert isinstance(config.testing, bool)
        assert isinstance(config.skip_external_services, bool)


class TestEnvironmentVariables:
    """Test environment variable handling - MICRO CHUNK 3B."""

    @staticmethod
    @patch.dict(os.environ, {"LOG_LEVEL": "ERROR"}, clear=False)
    def test_log_level_from_env() -> None:
        """Test setting log level from environment variable."""
        # Act
        config = Settings()

        # Assert
        assert config.log_level == "ERROR"

    @staticmethod
    @patch.dict(
        os.environ,
        {
            "ENVIRONMENT": "production",
            "SKIP_EXTERNAL_SERVICES": "true",
            "ENABLE_AUTH": "false",
        },
        clear=False,
    )
    def test_environment_from_env() -> None:
        """Test setting environment from environment variable."""
        # Act
        config = Settings()

        # Assert
        assert config.environment == "production"

    @staticmethod
    @patch.dict(os.environ, {"DEBUG": "true"}, clear=False)
    def test_debug_flag_from_env() -> None:
        """Test setting debug flag from environment variable."""
        # Act
        config = Settings()

        # Assert
        assert config.debug is True

    @staticmethod
    @patch.dict(os.environ, {"DEBUG": "false"}, clear=False)
    def test_debug_flag_false_from_env() -> None:
        """Test setting debug flag to false from environment variable."""
        # Act
        config = Settings()

        # Assert
        assert config.debug is False


class TestConfigValidation:
    """Test configuration validation - MICRO CHUNK 3C."""

    @staticmethod
    def test_valid_log_levels() -> None:
        """Test validation of log level values."""
        # Arrange
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        # Act & Assert
        for level in valid_levels:
            with patch.dict(os.environ, {"LOG_LEVEL": level}):
                config = Settings()
                assert config.log_level == level

    @staticmethod
    def test_valid_environments() -> None:
        """Test validation of environment values."""
        # Arrange
        valid_envs = ["development", "production", "test"]

        # Act & Assert
        for env in valid_envs:
            # Add skip external services for production to avoid validation errors
            env_vars = {"ENVIRONMENT": env}
            if env == "production":
                env_vars["SKIP_EXTERNAL_SERVICES"] = "true"
                env_vars["ENABLE_AUTH"] = "false"
            with patch.dict(os.environ, env_vars):
                config = Settings()
                assert config.environment == env

    @staticmethod
    def test_boolean_string_conversion() -> None:
        """Test conversion of string values to boolean."""
        # Test various string representations of boolean values
        true_values = ["true", "True", "TRUE", "1", "yes", "Yes"]
        false_values = ["false", "False", "FALSE", "0", "no", "No"]

        for true_val in true_values:
            with patch.dict(os.environ, {"DEBUG": true_val}):
                config = Settings()
                assert config.debug is True

        for false_val in false_values:
            with patch.dict(os.environ, {"DEBUG": false_val}):
                config = Settings()
                assert config.debug is False

    @staticmethod
    def test_settings_helper_methods() -> None:
        """Test Settings helper methods."""
        # Act
        config = Settings()

        # Assert helper methods exist and work
        assert isinstance(config.is_development(), bool)
        assert isinstance(config.is_production(), bool)
        assert isinstance(config.is_testing(), bool)
        assert isinstance(config.should_use_mock_services(), bool)


class TestConfigErrorHandling:
    """Test configuration error handling - MICRO CHUNK 3D."""

    @staticmethod
    def test_missing_configuration_error_creation() -> None:
        """Test creating MissingConfigurationError."""
        # Arrange
        config_key = "MISSING_KEY"

        # Act
        error = MissingConfigurationError(config_key)

        # Assert
        assert "Missing required configuration: MISSING_KEY" in str(error)
        assert error.config_key == config_key

    @staticmethod
    def test_invalid_configuration_error_creation() -> None:
        """Test creating InvalidConfigurationError."""
        # Arrange
        config_key = "INVALID_KEY"
        value = "invalid_value"
        reason = "not in allowed values"

        # Act
        error = InvalidConfigurationError(config_key, value, reason)

        # Assert
        assert config_key in str(error)
        assert str(value) in str(error)
        assert reason in str(error)
        assert error.config_key == config_key
        assert error.value == value
        assert error.reason == reason

    @staticmethod
    def test_configuration_error_base_class() -> None:
        """Test ConfigurationError base class."""
        # Arrange
        config_key = "TEST_KEY"

        # Act
        error = MissingConfigurationError(config_key)

        # Assert
        assert config_key in str(error)
        assert error.config_key == config_key

    @staticmethod
    def test_configuration_error_inheritance() -> None:
        """Test configuration error inheritance chain."""
        # Arrange & Act
        missing_error = MissingConfigurationError("TEST_KEY")
        invalid_error = InvalidConfigurationError("TEST_KEY", "value", "reason")

        # Assert
        assert isinstance(missing_error, Exception)
        assert isinstance(invalid_error, Exception)


class TestConfigDefaults:
    """Test configuration default values - MICRO CHUNK 3E."""

    @staticmethod
    def test_default_port_value() -> None:
        """Test default port configuration."""
        # Act
        config = Settings()

        # Assert
        assert hasattr(config, "port")
        assert isinstance(config.port, int)
        assert config.port > 0

    @staticmethod
    def test_default_host_value() -> None:
        """Test default host configuration."""
        # Act
        config = Settings()

        # Assert
        assert hasattr(config, "host")
        assert isinstance(config.host, str)
        assert len(config.host) > 0

    @staticmethod
    def test_default_cors_origins() -> None:
        """Test default CORS origins configuration."""
        # Act
        config = Settings()

        # Assert
        if hasattr(config, "cors_origins"):
            assert isinstance(config.cors_origins, list)

    @staticmethod
    def test_config_string_representation() -> None:
        """Test string representation of config object."""
        # Act
        config = Settings()
        config_str = str(config)

        # Assert
        assert isinstance(config_str, str)
        assert len(config_str) > 0

    @staticmethod
    def test_get_settings_function() -> None:
        """Test get_settings cached function."""
        # Act
        settings1 = get_settings()
        settings2 = get_settings()

        # Assert - Should return same cached instance
        assert settings1 is settings2
        assert isinstance(settings1, Settings)

    @staticmethod
    def test_middleware_config_creation() -> None:
        """Test MiddlewareConfig creation."""
        # Act
        middleware_config = MiddlewareConfig()

        # Assert
        assert isinstance(middleware_config.enabled, bool)
        assert isinstance(middleware_config.cache_enabled, bool)
        assert isinstance(middleware_config.cache_ttl_seconds, int)
        assert isinstance(middleware_config.exempt_paths, list)
