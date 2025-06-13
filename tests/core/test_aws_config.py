"""Tests for AWS configuration module."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from clarity.core.aws_config import AWSConfig


class TestAWSConfigBasics:
    """Test AWS configuration basic functionality."""

    def test_aws_config_creation_with_defaults(self):
        """Test AWSConfig creation with default values."""
        config = AWSConfig()
        
        # Test default values
        assert config.aws_region == "us-east-1"
        assert config.aws_access_key_id is None
        assert config.aws_secret_access_key is None
        assert config.cognito_user_pool_id is None
        assert config.cognito_client_id is None
        assert config.cognito_region is None
        assert config.dynamodb_table_name == "clarity-health-data"
        assert config.dynamodb_endpoint_url is None
        assert config.s3_bucket_name == "clarity-health-uploads"
        assert config.s3_endpoint_url is None
        assert config.sqs_queue_url is None
        assert config.sns_topic_arn is None
        assert config.gemini_api_key is None
        assert config.gemini_model == "gemini-1.5-flash"

    def test_aws_config_creation_with_values(self):
        """Test AWSConfig creation with explicit values."""
        config = AWSConfig(
            aws_region="us-west-2",
            aws_access_key_id="test-access-key",
            aws_secret_access_key="test-secret-key",
            cognito_user_pool_id="us-west-2_test123",
            cognito_client_id="test-client-id",
            cognito_region="us-west-2",
            dynamodb_table_name="test-table",
            dynamodb_endpoint_url="http://localhost:8000",
            s3_bucket_name="test-bucket",
            s3_endpoint_url="http://localhost:9000",
            sqs_queue_url="https://sqs.us-west-2.amazonaws.com/123456789012/test-queue",
            sns_topic_arn="arn:aws:sns:us-west-2:123456789012:test-topic",
            gemini_api_key="test-gemini-key",
            gemini_model="gemini-pro"
        )
        
        assert config.aws_region == "us-west-2"
        assert config.aws_access_key_id == "test-access-key"
        assert config.aws_secret_access_key == "test-secret-key"
        assert config.cognito_user_pool_id == "us-west-2_test123"
        assert config.cognito_client_id == "test-client-id"
        assert config.cognito_region == "us-west-2"
        assert config.dynamodb_table_name == "test-table"
        assert config.dynamodb_endpoint_url == "http://localhost:8000"
        assert config.s3_bucket_name == "test-bucket"
        assert config.s3_endpoint_url == "http://localhost:9000"
        assert config.sqs_queue_url == "https://sqs.us-west-2.amazonaws.com/123456789012/test-queue"
        assert config.sns_topic_arn == "arn:aws:sns:us-west-2:123456789012:test-topic"
        assert config.gemini_api_key == "test-gemini-key"
        assert config.gemini_model == "gemini-pro"

    def test_aws_config_field_descriptions(self):
        """Test that all fields have proper descriptions."""
        # This tests the Field configurations
        fields = AWSConfig.model_fields
        
        assert "AWS region" in fields["aws_region"].description
        assert "AWS access key ID" in fields["aws_access_key_id"].description
        assert "AWS secret access key" in fields["aws_secret_access_key"].description
        assert "Cognito user pool ID" in fields["cognito_user_pool_id"].description
        assert "Cognito app client ID" in fields["cognito_client_id"].description
        assert "Cognito region" in fields["cognito_region"].description
        assert "DynamoDB table name" in fields["dynamodb_table_name"].description
        assert "DynamoDB endpoint" in fields["dynamodb_endpoint_url"].description
        assert "S3 bucket" in fields["s3_bucket_name"].description
        assert "S3 endpoint" in fields["s3_endpoint_url"].description
        assert "SQS queue URL" in fields["sqs_queue_url"].description
        assert "SNS topic ARN" in fields["sns_topic_arn"].description
        assert "Google Gemini API key" in fields["gemini_api_key"].description
        assert "Gemini model" in fields["gemini_model"].description


class TestAWSConfigEnvironmentLoading:
    """Test AWS configuration loading from environment variables."""

    def test_config_loads_from_environment(self):
        """Test that config loads values from environment variables."""
        env_vars = {
            "AWS_REGION": "eu-west-1",
            "AWS_ACCESS_KEY_ID": "env-access-key",
            "AWS_SECRET_ACCESS_KEY": "env-secret-key",
            "COGNITO_USER_POOL_ID": "eu-west-1_env123",
            "COGNITO_CLIENT_ID": "env-client-id",
            "COGNITO_REGION": "eu-west-1",
            "DYNAMODB_TABLE_NAME": "env-table",
            "DYNAMODB_ENDPOINT_URL": "http://localhost:8001",
            "S3_BUCKET_NAME": "env-bucket",
            "S3_ENDPOINT_URL": "http://localhost:9001",
            "SQS_QUEUE_URL": "https://sqs.eu-west-1.amazonaws.com/123456789012/env-queue",
            "SNS_TOPIC_ARN": "arn:aws:sns:eu-west-1:123456789012:env-topic",
            "GEMINI_API_KEY": "env-gemini-key",
            "GEMINI_MODEL": "gemini-1.5-pro"
        }
        
        with patch.dict(os.environ, env_vars):
            config = AWSConfig()
            
            assert config.aws_region == "eu-west-1"
            assert config.aws_access_key_id == "env-access-key"
            assert config.aws_secret_access_key == "env-secret-key"
            assert config.cognito_user_pool_id == "eu-west-1_env123"
            assert config.cognito_client_id == "env-client-id"
            assert config.cognito_region == "eu-west-1"
            assert config.dynamodb_table_name == "env-table"
            assert config.dynamodb_endpoint_url == "http://localhost:8001"
            assert config.s3_bucket_name == "env-bucket"
            assert config.s3_endpoint_url == "http://localhost:9001"
            assert config.sqs_queue_url == "https://sqs.eu-west-1.amazonaws.com/123456789012/env-queue"
            assert config.sns_topic_arn == "arn:aws:sns:eu-west-1:123456789012:env-topic"
            assert config.gemini_api_key == "env-gemini-key"
            assert config.gemini_model == "gemini-1.5-pro"

    def test_config_explicit_values_override_env(self):
        """Test that explicit values override environment variables."""
        env_vars = {
            "AWS_REGION": "eu-west-1",
            "GEMINI_MODEL": "gemini-1.5-pro"
        }
        
        with patch.dict(os.environ, env_vars):
            config = AWSConfig(
                aws_region="ap-southeast-1",
                gemini_model="gemini-pro"
            )
            
            # Explicit values should override environment
            assert config.aws_region == "ap-southeast-1"
            assert config.gemini_model == "gemini-pro"

    def test_config_partial_env_loading(self):
        """Test config with some values from env and some defaults."""
        env_vars = {
            "AWS_REGION": "ca-central-1",
            "COGNITO_USER_POOL_ID": "ca-central-1_partial123",
            "GEMINI_API_KEY": "partial-gemini-key"
        }
        
        with patch.dict(os.environ, env_vars):
            config = AWSConfig()
            
            # From environment
            assert config.aws_region == "ca-central-1"
            assert config.cognito_user_pool_id == "ca-central-1_partial123"
            assert config.gemini_api_key == "partial-gemini-key"
            
            # Defaults for others
            assert config.aws_access_key_id is None
            assert config.dynamodb_table_name == "clarity-health-data"
            assert config.gemini_model == "gemini-1.5-flash"


class TestAWSConfigStringRepresentation:
    """Test AWS configuration string representation and serialization."""

    def test_config_model_dump(self):
        """Test that config can be serialized to dict."""
        config = AWSConfig(
            aws_region="us-west-1",
            dynamodb_table_name="test-dump-table"
        )
        
        config_dict = config.model_dump()
        
        assert isinstance(config_dict, dict)
        assert config_dict["aws_region"] == "us-west-1"
        assert config_dict["dynamodb_table_name"] == "test-dump-table"
        assert "aws_access_key_id" in config_dict

    def test_config_model_dump_exclude_none(self):
        """Test config serialization excluding None values."""
        config = AWSConfig(aws_region="us-west-1")
        
        config_dict = config.model_dump(exclude_none=True)
        
        assert "aws_region" in config_dict
        assert "dynamodb_table_name" in config_dict  # Has default value
        assert "gemini_model" in config_dict  # Has default value
        assert "aws_access_key_id" not in config_dict  # None value excluded

    def test_config_json_serialization(self):
        """Test that config can be serialized to JSON."""
        config = AWSConfig(aws_region="us-west-1", gemini_api_key="test-key")
        
        json_str = config.model_dump_json()
        
        assert isinstance(json_str, str)
        assert "us-west-1" in json_str
        assert "test-key" in json_str


class TestAWSConfigValidation:
    """Test AWS configuration validation."""

    def test_config_accepts_none_values(self):
        """Test that config accepts None for optional fields."""
        config = AWSConfig(
            aws_access_key_id=None,
            aws_secret_access_key=None,
            cognito_user_pool_id=None,
            cognito_client_id=None,
            cognito_region=None,
            dynamodb_endpoint_url=None,
            s3_endpoint_url=None,
            sqs_queue_url=None,
            sns_topic_arn=None,
            gemini_api_key=None
        )
        
        # Should not raise any validation errors
        assert config.aws_access_key_id is None
        assert config.cognito_user_pool_id is None
        assert config.gemini_api_key is None

    def test_config_with_empty_strings(self):
        """Test that config handles empty strings."""
        config = AWSConfig(
            aws_region="",  # This should be allowed since it's str, not str | None
            dynamodb_table_name="",
            s3_bucket_name="",
            gemini_model=""
        )
        
        assert config.aws_region == ""
        assert config.dynamodb_table_name == ""
        assert config.s3_bucket_name == ""
        assert config.gemini_model == ""


class TestAWSConfigClassConfiguration:
    """Test AWS configuration class-level settings."""

    def test_config_class_has_env_file_setting(self):
        """Test that Config class has proper env_file setting."""
        config_class = AWSConfig.model_config
        
        assert config_class.get("env_file") == ".env"
        assert config_class.get("env_file_encoding") == "utf-8"

    def test_config_model_fields_exist(self):
        """Test that all expected model fields are defined."""
        fields = AWSConfig.model_fields
        
        expected_fields = [
            "aws_region", "aws_access_key_id", "aws_secret_access_key",
            "cognito_user_pool_id", "cognito_client_id", "cognito_region",
            "dynamodb_table_name", "dynamodb_endpoint_url",
            "s3_bucket_name", "s3_endpoint_url",
            "sqs_queue_url", "sns_topic_arn",
            "gemini_api_key", "gemini_model"
        ]
        
        for field in expected_fields:
            assert field in fields

    def test_config_default_values_type_safety(self):
        """Test that default values have correct types."""
        config = AWSConfig()
        
        assert isinstance(config.aws_region, str)
        assert isinstance(config.dynamodb_table_name, str)
        assert isinstance(config.s3_bucket_name, str)
        assert isinstance(config.gemini_model, str)
        
        # Optional fields should be None or str
        assert config.aws_access_key_id is None or isinstance(config.aws_access_key_id, str)
        assert config.cognito_region is None or isinstance(config.cognito_region, str) 