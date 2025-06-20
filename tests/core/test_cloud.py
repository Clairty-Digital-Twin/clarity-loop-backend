"""Tests for AWS cloud utility functions."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, Mock, patch

import boto3

from clarity.core.cloud import (
    gemini_api_key,
    get_aws_session,
    get_cognito_client,
    get_dynamodb_resource,
)


class TestAWSSessionCreation:
    """Test AWS session creation functionality."""

    def test_get_aws_session_default_region(self):
        """Test AWS session creation with default region."""
        session = get_aws_session()

        assert isinstance(session, boto3.Session)
        assert session.region_name == "us-east-1"

    def test_get_aws_session_custom_region(self):
        """Test AWS session creation with custom region."""
        session = get_aws_session(region_name="us-west-2")

        assert isinstance(session, boto3.Session)
        assert session.region_name == "us-west-2"

    def test_get_aws_session_various_regions(self):
        """Test AWS session creation with various valid regions."""
        test_regions = [
            "us-east-1",
            "us-west-1",
            "us-west-2",
            "eu-west-1",
            "eu-central-1",
            "ap-southeast-1",
            "ap-northeast-1",
        ]

        for region in test_regions:
            session = get_aws_session(region_name=region)
            assert isinstance(session, boto3.Session)
            assert session.region_name == region

    def test_get_aws_session_with_credentials_from_env(self):
        """Test AWS session creation with credentials from environment."""
        env_vars = {
            "AWS_ACCESS_KEY_ID": "test-access-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret-key",
        }

        with patch.dict(os.environ, env_vars):
            session = get_aws_session("us-east-1")

            # Session should be created successfully
            assert isinstance(session, boto3.Session)
            assert session.region_name == "us-east-1"


class TestCognitoClientCreation:
    """Test Cognito client creation functionality."""

    def test_get_cognito_client_default_region(self):
        """Test Cognito client creation with default region."""
        client = get_cognito_client()

        # Should return a Cognito IDP client
        assert hasattr(client, "create_user_pool")
        assert hasattr(client, "create_user_pool_client")
        assert hasattr(client, "admin_create_user")

    def test_get_cognito_client_custom_region(self):
        """Test Cognito client creation with custom region."""
        client = get_cognito_client(region_name="eu-west-1")

        # Should return a Cognito IDP client
        assert hasattr(client, "create_user_pool")
        assert hasattr(client, "list_user_pools")

    def test_get_cognito_client_methods_exist(self):
        """Test that Cognito client has expected methods."""
        client = get_cognito_client()

        # Test for common Cognito IDP methods
        expected_methods = [
            "admin_create_user",
            "admin_delete_user",
            "admin_get_user",
            "admin_initiate_auth",
            "admin_set_user_password",
            "create_user_pool",
            "create_user_pool_client",
            "list_user_pools",
            "sign_up",
            "confirm_sign_up",
            "initiate_auth",
            "respond_to_auth_challenge",
        ]

        for method in expected_methods:
            assert hasattr(client, method)

    def test_get_cognito_client_different_regions(self):
        """Test Cognito client creation with different regions."""
        regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]

        for region in regions:
            client = get_cognito_client(region_name=region)
            assert hasattr(client, "create_user_pool")


class TestDynamoDBResourceCreation:
    """Test DynamoDB resource creation functionality."""

    def test_get_dynamodb_resource_default_region(self):
        """Test DynamoDB resource creation with default region."""
        resource = get_dynamodb_resource()

        # Should return a DynamoDB resource
        assert hasattr(resource, "create_table")
        assert hasattr(resource, "Table")
        assert hasattr(resource, "tables")

    def test_get_dynamodb_resource_custom_region(self):
        """Test DynamoDB resource creation with custom region."""
        resource = get_dynamodb_resource(region_name="eu-central-1")

        # Should return a DynamoDB resource
        assert hasattr(resource, "create_table")
        assert hasattr(resource, "Table")

    def test_get_dynamodb_resource_methods_exist(self):
        """Test that DynamoDB resource has expected methods."""
        resource = get_dynamodb_resource()

        # Test for common DynamoDB resource methods
        expected_methods = ["create_table", "Table", "tables"]

        for method in expected_methods:
            assert hasattr(resource, method)

    def test_get_dynamodb_resource_different_regions(self):
        """Test DynamoDB resource creation with different regions."""
        regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]

        for region in regions:
            resource = get_dynamodb_resource(region_name=region)
            assert hasattr(resource, "create_table")


class TestGeminiAPIKey:
    """Test Gemini API key functionality."""

    def test_gemini_api_key_not_set(self):
        """Test gemini_api_key when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing GEMINI_API_KEY
            if "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]

            api_key = gemini_api_key()
            assert api_key is None

    def test_gemini_api_key_set(self):
        """Test gemini_api_key when environment variable is set."""
        test_key = "test-gemini-api-key-12345"

        with patch.dict(os.environ, {"GEMINI_API_KEY": test_key}):
            api_key = gemini_api_key()
            assert api_key == test_key

    def test_gemini_api_key_empty_string(self):
        """Test gemini_api_key when environment variable is empty string."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            api_key = gemini_api_key()
            assert not api_key

    def test_gemini_api_key_whitespace(self):
        """Test gemini_api_key when environment variable contains whitespace."""
        test_key = "  test-gemini-key-with-spaces  "

        with patch.dict(os.environ, {"GEMINI_API_KEY": test_key}):
            api_key = gemini_api_key()
            assert api_key == test_key  # Should preserve whitespace

    def test_gemini_api_key_different_values(self):
        """Test gemini_api_key with different valid values."""
        test_keys = [
            "simple-key",
            "AIzaSyC123456789abcdef",  # Typical Google API key format
            "key-with-dashes-and-numbers-123",
            "verylongapikeywithnospaces1234567890abcdefghijklmnop",
            "key_with_underscores_123",
        ]

        for test_key in test_keys:
            with patch.dict(os.environ, {"GEMINI_API_KEY": test_key}):
                api_key = gemini_api_key()
                assert api_key == test_key

    def test_gemini_api_key_os_environ_access(self):
        """Test that gemini_api_key correctly accesses os.environ."""
        with patch("clarity.core.cloud.os.environ.get") as mock_get:
            mock_get.return_value = "mocked-api-key"

            api_key = gemini_api_key()

            mock_get.assert_called_once_with("GEMINI_API_KEY")
            assert api_key == "mocked-api-key"


class TestCloudModuleIntegration:
    """Test integration scenarios for cloud module functions."""

    def test_all_functions_return_expected_types(self):
        """Test that all functions return expected types."""
        # AWS Session
        session = get_aws_session()
        assert isinstance(session, boto3.Session)

        # Gemini API key
        api_key = gemini_api_key()
        assert api_key is None or isinstance(api_key, str)

    def test_functions_work_independently(self):
        """Test that functions work independently with different parameters."""
        # Different regions for different services
        session_region = "us-east-1"
        cognito_region = "us-west-2"
        dynamodb_region = "eu-west-1"

        session = get_aws_session(region_name=session_region)
        assert session.region_name == session_region

        cognito_client = get_cognito_client(region_name=cognito_region)
        assert hasattr(cognito_client, "create_user_pool")

        dynamodb_resource = get_dynamodb_resource(region_name=dynamodb_region)
        assert hasattr(dynamodb_resource, "create_table")

    def test_environment_isolation(self):
        """Test that environment variables don't interfere between calls."""
        # Test with API key set
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-1"}):
            api_key_1 = gemini_api_key()
            assert api_key_1 == "test-key-1"

        # Test with different API key
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-2"}):
            api_key_2 = gemini_api_key()
            assert api_key_2 == "test-key-2"

        # Test with no API key
        with patch.dict(os.environ, {}, clear=True):
            if "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]
            api_key_3 = gemini_api_key()
            assert api_key_3 is None


class TestCloudModuleBoto3Integration:
    """Test that all functions properly use boto3."""

    @patch("clarity.core.cloud.boto3.Session")
    def test_get_aws_session_calls_boto3_session(
        self, mock_boto3_session: MagicMock
    ) -> None:
        """Test that get_aws_session properly calls boto3.Session."""
        mock_session = Mock()
        mock_boto3_session.return_value = mock_session

        result = get_aws_session(region_name="test-region")

        mock_boto3_session.assert_called_once_with(region_name="test-region")
        assert result == mock_session

    @patch("clarity.core.cloud.boto3.client")
    def test_get_cognito_client_calls_boto3_client(
        self, mock_boto3_client: MagicMock
    ) -> None:
        """Test that get_cognito_client properly calls boto3.client."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client

        result = get_cognito_client(region_name="test-region")

        mock_boto3_client.assert_called_once_with(
            "cognito-idp", region_name="test-region"
        )
        assert result == mock_client

    @patch("clarity.core.cloud.boto3.resource")
    def test_get_dynamodb_resource_calls_boto3_resource(
        self, mock_boto3_resource: MagicMock
    ) -> None:
        """Test that get_dynamodb_resource properly calls boto3.resource."""
        mock_resource = Mock()
        mock_boto3_resource.return_value = mock_resource

        result = get_dynamodb_resource(region_name="test-region")

        mock_boto3_resource.assert_called_once_with(
            "dynamodb", region_name="test-region"
        )
        assert result == mock_resource
