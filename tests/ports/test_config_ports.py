"""Test config ports module."""

from unittest.mock import Mock

import pytest

from clarity.ports.config_ports import IConfigProvider


class MockConfigProvider(IConfigProvider):
    """Mock implementation of IConfigProvider for testing."""

    def get_setting(self, key: str, *, default=None):
        return default

    def is_development(self) -> bool:
        return True

    def should_skip_external_services(self) -> bool:
        return False

    def is_auth_enabled(self) -> bool:
        return True

    def get_aws_region(self) -> str:
        return "us-east-1"

    def get_log_level(self) -> str:
        return "INFO"

    def get_middleware_config(self):
        return Mock()

    def get_settings_model(self):
        return Mock()


def test_iconfigprovider_abstract_methods():
    """Test that IConfigProvider is abstract."""
    # Cannot instantiate abstract class
    with pytest.raises(TypeError):
        IConfigProvider()


def test_get_dynamodb_url_not_implemented():
    """Test get_dynamodb_url raises NotImplementedError."""
    provider = MockConfigProvider()

    with pytest.raises(NotImplementedError):
        provider.get_dynamodb_url()


def test_mock_config_provider_implementation():
    """Test mock implementation works correctly."""
    provider = MockConfigProvider()

    assert provider.get_setting("test", default="value") == "value"
    assert provider.is_development() is True
    assert provider.should_skip_external_services() is False
    assert provider.is_auth_enabled() is True
    assert provider.get_aws_region() == "us-east-1"
    assert provider.get_log_level() == "INFO"
    assert provider.get_middleware_config() is not None
    assert provider.get_settings_model() is not None
