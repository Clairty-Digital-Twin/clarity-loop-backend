"""Tests for CLARITY configuration schema validation."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from clarity.startup.config_schema import ClarityConfig, Environment, load_config


class TestClarityConfig:
    """Test configuration schema validation."""

    def test_default_configuration(self) -> None:
        """Test default configuration loads successfully."""
        with patch.dict(os.environ, {}, clear=True):
            config, errors = ClarityConfig.validate_from_env()

            assert config is not None
            assert len(errors) == 0
            assert config.environment == Environment.DEVELOPMENT
            assert config.aws.region == "us-east-1"
            assert config.port == 8000

    def test_environment_variable_parsing(self) -> None:
        """Test environment variable parsing."""
        env_vars = {
            "ENVIRONMENT": "production",
            "AWS_REGION": "us-east-1",
            "PORT": "8080",
            "ENABLE_AUTH": "false",
            "DEBUG": "true",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config, errors = ClarityConfig.validate_from_env()

            assert config is not None
            assert len(errors) == 0
            assert config.environment == Environment.PRODUCTION
            assert config.aws.region == "us-east-1"
            assert config.port == 8080
            assert config.enable_auth is False
            assert config.debug is True

    def test_nested_configuration_parsing(self) -> None:
        """Test nested configuration from environment variables."""
        env_vars = {
            "AWS_REGION": "eu-west-1",
            "COGNITO_USER_POOL_ID": "eu-west-1_test123",
            "COGNITO_CLIENT_ID": "test-client-id-12345",
            "DYNAMODB_TABLE_NAME": "test-table",
            "S3_BUCKET_NAME": "test-bucket",
            "GEMINI_API_KEY": "test-gemini-key",
            "SECRET_KEY": "test-secret-key-12345",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config, errors = ClarityConfig.validate_from_env()

            assert config is not None
            assert len(errors) == 0
            assert config.aws.region == "eu-west-1"
            assert config.cognito.user_pool_id == "eu-west-1_test123"
            assert config.cognito.client_id == "test-client-id-12345"
            assert config.dynamodb.table_name == "test-table"
            assert config.s3.bucket_name == "test-bucket"
            assert config.gemini.api_key == "test-gemini-key"
            assert config.security.secret_key == "test-secret-key-12345"

    def test_production_validation_success(self) -> None:
        """Test successful production environment validation."""
        env_vars = {
            "ENVIRONMENT": "production",
            "SECRET_KEY": "production-secret-key-with-sufficient-length",
            "COGNITO_USER_POOL_ID": "us-east-1_validpool",
            "COGNITO_CLIENT_ID": "valid-client-id-12345678",
            "CORS_ALLOWED_ORIGINS": "https://app.example.com,https://api.example.com",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config, errors = ClarityConfig.validate_from_env()

            assert config is not None
            assert len(errors) == 0
            assert config.is_production()

    def test_production_validation_failures(self) -> None:
        """Test production environment validation failures."""
        env_vars = {
            "ENVIRONMENT": "production",
            "ENABLE_AUTH": "true",
            # Missing required production fields
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config, errors = ClarityConfig.validate_from_env()

            assert config is None
            assert len(errors) > 0

            # Check for specific production errors
            error_text = " ".join(errors)
            assert "COGNITO_USER_POOL_ID" in error_text
            assert "COGNITO_CLIENT_ID" in error_text

    def test_invalid_values_validation(self) -> None:
        """Test validation of invalid configuration values."""
        test_cases = [
            {
                "env": {"PORT": "99999"},  # Invalid port
                "should_fail": True,
            },
            {
                "env": {"AWS_REGION": "invalid-region"},  # Invalid AWS region
                "should_fail": True,
            },
            {
                "env": {"COGNITO_USER_POOL_ID": "invalid"},  # Invalid pool ID format
                "should_fail": True,
            },
            {
                "env": {
                    "CORS_ALLOWED_ORIGINS": "not-a-url,also-not-a-url"
                },  # Invalid URLs
                "should_fail": True,
            },
        ]

        for case in test_cases:
            with patch.dict(os.environ, case["env"], clear=True):
                config, errors = ClarityConfig.validate_from_env()

                if case["should_fail"]:
                    assert config is None or len(errors) > 0
                else:
                    assert config is not None
                    assert len(errors) == 0

    def test_cors_wildcard_validation(self) -> None:
        """Test CORS wildcard validation."""
        # Partial wildcards should fail
        env_vars = {
            "CORS_ALLOWED_ORIGINS": "https://*.example.com",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config, errors = ClarityConfig.validate_from_env()

            assert config is None or len(errors) > 0

    def test_service_requirements(self) -> None:
        """Test service requirements calculation."""
        # Test with auth enabled
        env_vars = {
            "ENABLE_AUTH": "true",
            "SKIP_EXTERNAL_SERVICES": "false",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config, errors = ClarityConfig.validate_from_env()
            assert config is not None

            requirements = config.get_service_requirements()
            assert requirements["cognito"] is True
            assert requirements["dynamodb"] is True
            assert requirements["s3"] is True

        # Test with mock services
        env_vars = {
            "SKIP_EXTERNAL_SERVICES": "true",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config, errors = ClarityConfig.validate_from_env()
            assert config is not None

            requirements = config.get_service_requirements()
            assert requirements["cognito"] is False
            assert requirements["dynamodb"] is False
            assert requirements["s3"] is False

    def test_startup_summary(self) -> None:
        """Test startup summary generation."""
        env_vars = {
            "ENVIRONMENT": "development",
            "ENABLE_AUTH": "true",
            "AWS_REGION": "us-east-1",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config, errors = ClarityConfig.validate_from_env()
            assert config is not None

            summary = config.get_startup_summary()

            assert summary["environment"] == "development"
            assert summary["auth_enabled"] is True
            assert summary["aws_region"] == "us-east-1"
            assert "required_services" in summary
            assert "startup_timeout" in summary

    def test_should_use_mock_services(self) -> None:
        """Test mock services determination."""
        # Development should use mocks by default
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
            config, _ = ClarityConfig.validate_from_env()
            assert config is not None
            assert config.should_use_mock_services() is True

        # Testing should use mocks
        with patch.dict(os.environ, {"ENVIRONMENT": "testing"}, clear=True):
            config, _ = ClarityConfig.validate_from_env()
            assert config is not None
            assert config.should_use_mock_services() is True

        # Explicit skip should use mocks
        with patch.dict(os.environ, {"SKIP_EXTERNAL_SERVICES": "true"}, clear=True):
            config, _ = ClarityConfig.validate_from_env()
            assert config is not None
            assert config.should_use_mock_services() is True


class TestLoadConfig:
    """Test configuration loading function."""

    def test_load_config_success(self) -> None:
        """Test successful configuration loading."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
            config = load_config()
            assert config is not None
            assert config.environment == Environment.DEVELOPMENT

    def test_load_config_validation_error(self) -> None:
        """Test configuration loading with validation errors."""
        env_vars = {
            "ENVIRONMENT": "production",
            "ENABLE_AUTH": "true",
            # Missing required production settings
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValueError, match="Configuration validation failed"):
                load_config()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
