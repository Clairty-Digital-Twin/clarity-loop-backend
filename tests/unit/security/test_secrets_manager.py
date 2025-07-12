"""Unit tests for the secrets manager module."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from clarity.security.secrets_manager import (
    SecretsManager,
    get_secrets_manager,
    with_secret,
)


class TestSecretsManager:
    """Test cases for SecretsManager class."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        with patch("boto3.client") as mock_boto:
            manager = SecretsManager()
            assert manager.ssm_prefix == "/clarity/production"
            assert manager.region == "us-east-1"
            assert manager.cache_ttl == 300
            assert manager.use_ssm is False  # Default when not in AWS

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        manager = SecretsManager(
            ssm_prefix="/custom/prefix",
            region="eu-west-1",
            cache_ttl_seconds=600,
            use_ssm=False,
        )
        assert manager.ssm_prefix == "/custom/prefix"
        assert manager.region == "eu-west-1"
        assert manager.cache_ttl == 600
        assert manager.use_ssm is False

    def test_init_from_environment(self):
        """Test initialization from environment variables."""
        with patch.dict(
            os.environ,
            {
                "CLARITY_SSM_PREFIX": "/env/prefix",
                "AWS_DEFAULT_REGION": "ap-south-1",
                "CLARITY_SECRETS_CACHE_TTL": "120",
                "CLARITY_USE_SSM": "true",
            },
        ):
            with patch("boto3.client") as mock_boto:
                manager = SecretsManager()
                assert manager.ssm_prefix == "/env/prefix"
                assert manager.region == "ap-south-1"
                assert manager.cache_ttl == 120

    def test_get_string_from_env(self):
        """Test getting string value from environment variable."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            manager = SecretsManager(use_ssm=False)
            value = manager.get_string("test_key", env_var="TEST_VAR")
            assert value == "test_value"

    def test_get_string_from_default_env(self):
        """Test getting string value from default environment variable."""
        with patch.dict(os.environ, {"CLARITY_TEST_KEY": "default_value"}):
            manager = SecretsManager(use_ssm=False)
            value = manager.get_string("test_key")
            assert value == "default_value"

    def test_get_string_with_default(self):
        """Test getting string value with default."""
        manager = SecretsManager(use_ssm=False)
        value = manager.get_string("nonexistent_key", default="fallback")
        assert value == "fallback"

    @patch("boto3.client")
    def test_get_string_from_ssm(self, mock_boto):
        """Test getting string value from SSM."""
        mock_ssm = MagicMock()
        mock_boto.return_value = mock_ssm
        mock_ssm.get_parameter.return_value = {
            "Parameter": {"Value": "ssm_value"}
        }

        manager = SecretsManager(use_ssm=True)
        manager._ssm_client = mock_ssm
        
        value = manager.get_string("test_key")
        assert value == "ssm_value"
        
        mock_ssm.get_parameter.assert_called_with(
            Name="/clarity/production/test_key",
            WithDecryption=True,
        )

    @patch("boto3.client")
    def test_get_string_ssm_not_found(self, mock_boto):
        """Test handling SSM parameter not found."""
        mock_ssm = MagicMock()
        mock_boto.return_value = mock_ssm
        mock_ssm.exceptions.ParameterNotFound = ClientError
        mock_ssm.get_parameter.side_effect = ClientError(
            {"Error": {"Code": "ParameterNotFound"}},
            "GetParameter",
        )

        manager = SecretsManager(use_ssm=True)
        manager._ssm_client = mock_ssm
        
        value = manager.get_string("test_key", default="default")
        assert value == "default"

    def test_get_json(self):
        """Test getting JSON value."""
        test_json = {"key": "value", "number": 42}
        with patch.dict(
            os.environ,
            {"TEST_JSON": json.dumps(test_json)},
        ):
            manager = SecretsManager(use_ssm=False)
            value = manager.get_json("test_key", env_var="TEST_JSON")
            assert value == test_json

    def test_get_json_invalid(self):
        """Test handling invalid JSON."""
        with patch.dict(os.environ, {"TEST_JSON": "invalid json"}):
            manager = SecretsManager(use_ssm=False)
            value = manager.get_json(
                "test_key",
                env_var="TEST_JSON",
                default={"default": True},
            )
            assert value == {"default": True}

    def test_get_model_signature_key(self):
        """Test getting model signature key."""
        with patch.dict(
            os.environ,
            {"MODEL_SIGNATURE_KEY": "custom_signature_key"},
        ):
            manager = SecretsManager(use_ssm=False)
            key = manager.get_model_signature_key()
            assert key == "custom_signature_key"

    def test_get_model_signature_key_default(self):
        """Test getting model signature key with default."""
        manager = SecretsManager(use_ssm=False)
        key = manager.get_model_signature_key()
        assert key == "pat_model_integrity_key_2025"

    def test_get_model_checksums(self):
        """Test getting model checksums."""
        checksums = {
            "small": "checksum1",
            "medium": "checksum2",
            "large": "checksum3",
        }
        with patch.dict(
            os.environ,
            {"EXPECTED_MODEL_CHECKSUMS": json.dumps(checksums)},
        ):
            manager = SecretsManager(use_ssm=False)
            result = manager.get_model_checksums()
            assert result == checksums

    def test_get_model_checksums_default(self):
        """Test getting model checksums with defaults."""
        manager = SecretsManager(use_ssm=False)
        checksums = manager.get_model_checksums()
        assert "small" in checksums
        assert "medium" in checksums
        assert "large" in checksums

    def test_cache_functionality(self):
        """Test caching functionality."""
        manager = SecretsManager(use_ssm=False, cache_ttl_seconds=1)
        
        with patch.dict(os.environ, {"CLARITY_TEST_KEY": "cached_value"}):
            # First call should hit environment
            value1 = manager.get_string("test_key")
            assert value1 == "cached_value"
            
            # Change environment - cached value should be returned
            with patch.dict(os.environ, {"CLARITY_TEST_KEY": "new_value"}):
                value2 = manager.get_string("test_key")
                assert value2 == "cached_value"  # From cache
                
                # Wait for cache to expire
                import time
                time.sleep(1.1)
                
                value3 = manager.get_string("test_key")
                assert value3 == "new_value"  # Cache expired

    def test_refresh_cache(self):
        """Test cache refresh functionality."""
        manager = SecretsManager(use_ssm=False)
        
        with patch.dict(os.environ, {"CLARITY_TEST_KEY": "value1"}):
            value1 = manager.get_string("test_key")
            assert value1 == "value1"
            
            # Refresh specific key
            manager.refresh_cache("test_key")
            
            with patch.dict(os.environ, {"CLARITY_TEST_KEY": "value2"}):
                value2 = manager.get_string("test_key")
                assert value2 == "value2"  # Not from cache

    def test_refresh_all_cache(self):
        """Test refreshing entire cache."""
        manager = SecretsManager(use_ssm=False)
        
        with patch.dict(
            os.environ,
            {
                "CLARITY_KEY1": "value1",
                "CLARITY_KEY2": "value2",
            },
        ):
            # Populate cache
            manager.get_string("key1")
            manager.get_string("key2")
            
            # Clear all cache
            manager.refresh_cache()
            
            # Verify cache is empty
            assert len(manager._cache) == 0

    def test_health_check_ssm_disabled(self):
        """Test health check when SSM is disabled."""
        manager = SecretsManager(use_ssm=False)
        health = manager.health_check()
        
        assert health["service"] == "SecretsManager"
        assert health["use_ssm"] is False
        assert health["ssm_status"] == "disabled"

    @patch("boto3.client")
    def test_health_check_ssm_healthy(self, mock_boto):
        """Test health check when SSM is healthy."""
        mock_ssm = MagicMock()
        mock_boto.return_value = mock_ssm
        
        manager = SecretsManager(use_ssm=True)
        manager._ssm_client = mock_ssm
        
        health = manager.health_check()
        assert health["ssm_status"] == "healthy"

    @patch("boto3.client")
    def test_health_check_ssm_unhealthy(self, mock_boto):
        """Test health check when SSM is unhealthy."""
        mock_ssm = MagicMock()
        mock_boto.return_value = mock_ssm
        mock_ssm.describe_parameters.side_effect = Exception("Connection error")
        
        manager = SecretsManager(use_ssm=True)
        manager._ssm_client = mock_ssm
        
        health = manager.health_check()
        assert "unhealthy" in health["ssm_status"]

    def test_get_secrets_manager_singleton(self):
        """Test get_secrets_manager returns singleton."""
        manager1 = get_secrets_manager()
        manager2 = get_secrets_manager()
        assert manager1 is manager2

    def test_with_secret_decorator_string(self):
        """Test with_secret decorator for string values."""
        with patch.dict(os.environ, {"CLARITY_API_KEY": "secret123"}):
            @with_secret("api_key", str)
            def test_func(api_key=None):
                return api_key
            
            result = test_func()
            assert result == "secret123"

    def test_with_secret_decorator_json(self):
        """Test with_secret decorator for JSON values."""
        config = {"host": "localhost", "port": 5432}
        with patch.dict(
            os.environ,
            {"CLARITY_DB_CONFIG": json.dumps(config)},
        ):
            @with_secret("db_config", dict)
            def test_func(db_config=None):
                return db_config
            
            result = test_func()
            assert result == config

    @patch("boto3.client")
    def test_retry_logic(self, mock_boto):
        """Test retry logic for transient SSM failures."""
        mock_ssm = MagicMock()
        mock_boto.return_value = mock_ssm
        
        # First two calls fail, third succeeds
        mock_ssm.get_parameter.side_effect = [
            ClientError({"Error": {"Code": "ThrottlingException"}}, "GetParameter"),
            ClientError({"Error": {"Code": "ThrottlingException"}}, "GetParameter"),
            {"Parameter": {"Value": "success"}},
        ]
        
        manager = SecretsManager(use_ssm=True)
        manager._ssm_client = mock_ssm
        
        value = manager.get_string("test_key")
        assert value == "success"
        assert mock_ssm.get_parameter.call_count == 3