"""MICRO-FOCUSED AWS Config Tests.

🚀 COMPREHENSIVE TESTS FOR AWS CONFIG MODULE 🚀
Target: aws_config.py 0% → 95%+

Breaking down into MICRO pieces:
- Basic config creation with defaults
- Environment variable handling for all AWS settings
- Pydantic model validation
- Field defaults and descriptions
- Config class settings

FOCUSED ON COMPLETE COVERAGE!
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from clarity.core.aws_config import AWSConfig


class TestBasicAWSConfigCreation:
    """Test basic AWSConfig object creation - MICRO CHUNK 1."""

    @staticmethod
    def test_aws_config_creation_with_defaults() -> None:
        """Test creating AWSConfig with default values."""
        # Act
        config = AWSConfig()

        # Assert
        assert config is not None
        assert isinstance(config, AWSConfig)

    @staticmethod
    def test_aws_config_has_required_attributes() -> None:
        """Test AWSConfig has all required attributes."""
        # Act
        config = AWSConfig()

        # Assert - AWS Core Settings
        assert hasattr(config, "aws_region")
        assert hasattr(config, "aws_access_key_id")
        assert hasattr(config, "aws_secret_access_key")

        # Assert - Cognito Settings
        assert hasattr(config, "cognito_user_pool_id")
        assert hasattr(config, "cognito_client_id")
        assert hasattr(config, "cognito_region")

        # Assert - DynamoDB Settings
        assert hasattr(config, "dynamodb_table_name")
        assert hasattr(config, "dynamodb_endpoint_url")

        # Assert - S3 Settings
        assert hasattr(config, "s3_bucket_name")
        assert hasattr(config, "s3_endpoint_url")

        # Assert - SQS/SNS Settings
        assert hasattr(config, "sqs_queue_url")
        assert hasattr(config, "sns_topic_arn")

        # Assert - Gemini Settings
        assert hasattr(config, "gemini_api_key")
        assert hasattr(config, "gemini_model")

    @staticmethod
    @patch.dict(os.environ, {}, clear=True)
    def test_aws_config_default_values() -> None:
        """Test AWSConfig default values are correct."""
        # Act
        config = AWSConfig()

        # Assert - AWS Core Settings defaults
        assert config.aws_region == "us-east-1"
        assert config.aws_access_key_id is None
        assert config.aws_secret_access_key is None

        # Assert - Cognito Settings defaults
        assert config.cognito_user_pool_id is None
        assert config.cognito_client_id is None
        assert config.cognito_region is None

        # Assert - DynamoDB Settings defaults
        assert config.dynamodb_table_name == "clarity-health-data"
        assert config.dynamodb_endpoint_url is None

        # Assert - S3 Settings defaults
        assert config.s3_bucket_name == "clarity-health-uploads"
        assert config.s3_endpoint_url is None

        # Assert - SQS/SNS Settings defaults
        assert config.sqs_queue_url is None
        assert config.sns_topic_arn is None

        # Assert - Gemini Settings defaults
        assert config.gemini_api_key is None
        assert config.gemini_model == "gemini-1.5-flash"


class TestAWSCoreSettings:
    """Test AWS core settings configuration - MICRO CHUNK 2."""

    @staticmethod
    @patch.dict(os.environ, {"AWS_REGION": "us-west-2"}, clear=False)
    def test_aws_region_from_env() -> None:
        """Test setting AWS region from environment variable."""
        # Act
        config = AWSConfig()

        # Assert
        assert config.aws_region == "us-west-2"

    @staticmethod
    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test-access-key"}, clear=False)
    def test_aws_access_key_id_from_env() -> None:
        """Test setting AWS access key ID from environment variable."""
        # Act
        config = AWSConfig()

        # Assert
        assert config.aws_access_key_id == "test-access-key"

    @staticmethod
    @patch.dict(os.environ, {"AWS_SECRET_ACCESS_KEY": "test-secret-key"}, clear=False)
    def test_aws_secret_access_key_from_env() -> None:
        """Test setting AWS secret access key from environment variable."""
        # Act
        config = AWSConfig()

        # Assert
        assert config.aws_secret_access_key == "test-secret-key"

    @staticmethod
    @patch.dict(
        os.environ,
        {
            "AWS_REGION": "eu-west-1",
            "AWS_ACCESS_KEY_ID": "eu-access-key",
            "AWS_SECRET_ACCESS_KEY": "eu-secret-key",
        },
        clear=False,
    )
    def test_all_aws_core_settings_from_env() -> None:
        """Test setting all AWS core settings from environment variables."""
        # Act
        config = AWSConfig()

        # Assert
        assert config.aws_region == "eu-west-1"
        assert config.aws_access_key_id == "eu-access-key"
        assert config.aws_secret_access_key == "eu-secret-key"


class TestCognitoSettings:
    """Test Cognito settings configuration - MICRO CHUNK 3."""

    @staticmethod
    @patch.dict(os.environ, {"COGNITO_USER_POOL_ID": "us-east-1_TEST123"}, clear=False)
    def test_cognito_user_pool_id_from_env() -> None:
        """Test setting Cognito user pool ID from environment variable."""
        # Act
        config = AWSConfig()

        # Assert
        assert config.cognito_user_pool_id == "us-east-1_TEST123"

    @staticmethod
    @patch.dict(os.environ, {"COGNITO_CLIENT_ID": "test-client-id-123"}, clear=False)
    def test_cognito_client_id_from_env() -> None:
        """Test setting Cognito client ID from environment variable."""
        # Act
        config = AWSConfig()

        # Assert
        assert config.cognito_client_id == "test-client-id-123"

    @staticmethod
    @patch.dict(os.environ, {"COGNITO_REGION": "us-west-2"}, clear=False)
    def test_cognito_region_from_env() -> None:
        """Test setting Cognito region from environment variable."""
        # Act
        config = AWSConfig()

        # Assert
        assert config.cognito_region == "us-west-2"

    @staticmethod
    @patch.dict(
        os.environ,
        {
            "COGNITO_USER_POOL_ID": "us-east-1_FULLTEST",
            "COGNITO_CLIENT_ID": "full-client-id",
            "COGNITO_REGION": "eu-central-1",
        },
        clear=False,
    )
    def test_all_cognito_settings_from_env() -> None:
        """Test setting all Cognito settings from environment variables."""
        # Act
        config = AWSConfig()

        # Assert
        assert config.cognito_user_pool_id == "us-east-1_FULLTEST"
        assert config.cognito_client_id == "full-client-id"
        assert config.cognito_region == "eu-central-1"


class TestDynamoDBSettings:
    """Test DynamoDB settings configuration - MICRO CHUNK 4."""

    @staticmethod
    @patch.dict(os.environ, {"DYNAMODB_TABLE_NAME": "custom-table-name"}, clear=False)
    def test_dynamodb_table_name_from_env() -> None:
        """Test setting DynamoDB table name from environment variable."""
        # Act
        config = AWSConfig()

        # Assert
        assert config.dynamodb_table_name == "custom-table-name"

    @staticmethod
    @patch.dict(
        os.environ, {"DYNAMODB_ENDPOINT_URL": "http://localhost:8000"}, clear=False
    )
    def test_dynamodb_endpoint_url_from_env() -> None:
        """Test setting DynamoDB endpoint URL from environment variable."""
        # Act
        config = AWSConfig()

        # Assert
        assert config.dynamodb_endpoint_url == "http://localhost:8000"

    @staticmethod
    @patch.dict(
        os.environ,
        {
            "DYNAMODB_TABLE_NAME": "test-health-data",
            "DYNAMODB_ENDPOINT_URL": "http://dynamodb.local:8000",
        },
        clear=False,
    )
    def test_all_dynamodb_settings_from_env() -> None:
        """Test setting all DynamoDB settings from environment variables."""
        # Act
        config = AWSConfig()

        # Assert
        assert config.dynamodb_table_name == "test-health-data"
        assert config.dynamodb_endpoint_url == "http://dynamodb.local:8000"


class TestS3Settings:
    """Test S3 settings configuration - MICRO CHUNK 5."""

    @staticmethod
    @patch.dict(os.environ, {"S3_BUCKET_NAME": "custom-bucket-name"}, clear=False)
    def test_s3_bucket_name_from_env() -> None:
        """Test setting S3 bucket name from environment variable."""
        # Act
        config = AWSConfig()

        # Assert
        assert config.s3_bucket_name == "custom-bucket-name"

    @staticmethod
    @patch.dict(os.environ, {"S3_ENDPOINT_URL": "http://localhost:9000"}, clear=False)
    def test_s3_endpoint_url_from_env() -> None:
        """Test setting S3 endpoint URL from environment variable."""
        # Act
        config = AWSConfig()

        # Assert
        assert config.s3_endpoint_url == "http://localhost:9000"

    @staticmethod
    @patch.dict(
        os.environ,
        {
            "S3_BUCKET_NAME": "test-uploads-bucket",
            "S3_ENDPOINT_URL": "http://minio.local:9000",
        },
        clear=False,
    )
    def test_all_s3_settings_from_env() -> None:
        """Test setting all S3 settings from environment variables."""
        # Act
        config = AWSConfig()

        # Assert
        assert config.s3_bucket_name == "test-uploads-bucket"
        assert config.s3_endpoint_url == "http://minio.local:9000"


class TestSQSSNSSettings:
    """Test SQS/SNS settings configuration - MICRO CHUNK 6."""

    @staticmethod
    @patch.dict(
        os.environ,
        {"SQS_QUEUE_URL": "https://sqs.us-east-1.amazonaws.com/123456789/test-queue"},
        clear=False,
    )
    def test_sqs_queue_url_from_env() -> None:
        """Test setting SQS queue URL from environment variable."""
        # Act
        config = AWSConfig()

        # Assert
        assert (
            config.sqs_queue_url
            == "https://sqs.us-east-1.amazonaws.com/123456789/test-queue"
        )

    @staticmethod
    @patch.dict(
        os.environ,
        {"SNS_TOPIC_ARN": "arn:aws:sns:us-east-1:123456789:test-topic"},
        clear=False,
    )
    def test_sns_topic_arn_from_env() -> None:
        """Test setting SNS topic ARN from environment variable."""
        # Act
        config = AWSConfig()

        # Assert
        assert config.sns_topic_arn == "arn:aws:sns:us-east-1:123456789:test-topic"

    @staticmethod
    @patch.dict(
        os.environ,
        {
            "SQS_QUEUE_URL": "https://sqs.eu-west-1.amazonaws.com/987654321/production-queue",
            "SNS_TOPIC_ARN": "arn:aws:sns:eu-west-1:987654321:production-topic",
        },
        clear=False,
    )
    def test_all_sqs_sns_settings_from_env() -> None:
        """Test setting all SQS/SNS settings from environment variables."""
        # Act
        config = AWSConfig()

        # Assert
        assert (
            config.sqs_queue_url
            == "https://sqs.eu-west-1.amazonaws.com/987654321/production-queue"
        )
        assert (
            config.sns_topic_arn == "arn:aws:sns:eu-west-1:987654321:production-topic"
        )


class TestGeminiSettings:
    """Test Gemini settings configuration - MICRO CHUNK 7."""

    @staticmethod
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-api-key-123"}, clear=False)
    def test_gemini_api_key_from_env() -> None:
        """Test setting Gemini API key from environment variable."""
        # Act
        config = AWSConfig()

        # Assert
        assert config.gemini_api_key == "test-gemini-api-key-123"

    @staticmethod
    @patch.dict(os.environ, {"GEMINI_MODEL": "gemini-1.5-pro"}, clear=False)
    def test_gemini_model_from_env() -> None:
        """Test setting Gemini model from environment variable."""
        # Act
        config = AWSConfig()

        # Assert
        assert config.gemini_model == "gemini-1.5-pro"

    @staticmethod
    @patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "production-gemini-key",
            "GEMINI_MODEL": "gemini-1.5-ultra",
        },
        clear=False,
    )
    def test_all_gemini_settings_from_env() -> None:
        """Test setting all Gemini settings from environment variables."""
        # Act
        config = AWSConfig()

        # Assert
        assert config.gemini_api_key == "production-gemini-key"
        assert config.gemini_model == "gemini-1.5-ultra"


class TestPydanticModelBehavior:
    """Test Pydantic model behavior and validation - MICRO CHUNK 8."""

    @staticmethod
    def test_model_fields_have_descriptions() -> None:
        """Test that model fields have proper descriptions."""
        # Act
        config = AWSConfig()
        fields = config.model_fields

        # Assert key fields have descriptions
        assert "AWS region" in str(fields["aws_region"].description)
        assert "AWS access key ID" in str(fields["aws_access_key_id"].description)
        assert "Cognito user pool ID" in str(fields["cognito_user_pool_id"].description)
        assert "DynamoDB table name" in str(fields["dynamodb_table_name"].description)
        assert "S3 bucket" in str(fields["s3_bucket_name"].description)
        assert "Gemini API key" in str(fields["gemini_api_key"].description)

    @staticmethod
    def test_model_config_settings() -> None:
        """Test Pydantic model Config settings."""
        # Act
        config = AWSConfig()

        # Assert Config class exists and has proper settings
        assert "env_file" in config.model_config
        assert config.model_config["env_file"] == ".env"
        assert config.model_config["env_file_encoding"] == "utf-8"

    @staticmethod
    def test_model_dict_export() -> None:
        """Test exporting model to dictionary."""
        # Arrange
        with patch.dict(
            os.environ,
            {
                "AWS_REGION": "test-region",
                "DYNAMODB_TABLE_NAME": "test-table",
                "GEMINI_MODEL": "test-model",
            },
            clear=False,
        ):
            config = AWSConfig()

        # Act
        config_dict = config.model_dump()

        # Assert
        assert isinstance(config_dict, dict)
        assert config_dict["aws_region"] == "test-region"
        assert config_dict["dynamodb_table_name"] == "test-table"
        assert config_dict["gemini_model"] == "test-model"

    @staticmethod
    def test_model_validation_with_direct_init() -> None:
        """Test model validation when initializing with direct parameters."""
        # Act
        config = AWSConfig(
            aws_region="direct-region",
            dynamodb_table_name="direct-table",
            s3_bucket_name="direct-bucket",
            gemini_model="direct-model",
        )

        # Assert
        assert config.aws_region == "direct-region"
        assert config.dynamodb_table_name == "direct-table"
        assert config.s3_bucket_name == "direct-bucket"
        assert config.gemini_model == "direct-model"


class TestCompleteAWSConfiguration:
    """Test complete AWS configuration scenarios - MICRO CHUNK 9."""

    @staticmethod
    @patch.dict(
        os.environ,
        {
            # AWS Core
            "AWS_REGION": "us-west-2",
            "AWS_ACCESS_KEY_ID": "AKIA1234567890",
            "AWS_SECRET_ACCESS_KEY": "secret123",
            # Cognito
            "COGNITO_USER_POOL_ID": "us-west-2_ABCDEF123",
            "COGNITO_CLIENT_ID": "client123",
            "COGNITO_REGION": "us-west-2",
            # DynamoDB
            "DYNAMODB_TABLE_NAME": "production-health-data",
            # S3
            "S3_BUCKET_NAME": "production-uploads",
            # SQS/SNS
            "SQS_QUEUE_URL": "https://sqs.us-west-2.amazonaws.com/123456789/prod-queue",
            "SNS_TOPIC_ARN": "arn:aws:sns:us-west-2:123456789:prod-topic",
            # Gemini
            "GEMINI_API_KEY": "prod-gemini-key",
            "GEMINI_MODEL": "gemini-1.5-pro",
        },
        clear=False,
    )
    def test_complete_production_aws_config() -> None:
        """Test complete production AWS configuration."""
        # Act
        config = AWSConfig()

        # Assert all values are set correctly
        assert config.aws_region == "us-west-2"
        assert config.aws_access_key_id == "AKIA1234567890"
        assert config.aws_secret_access_key == "secret123"
        assert config.cognito_user_pool_id == "us-west-2_ABCDEF123"
        assert config.cognito_client_id == "client123"
        assert config.cognito_region == "us-west-2"
        assert config.dynamodb_table_name == "production-health-data"
        assert config.s3_bucket_name == "production-uploads"
        assert (
            config.sqs_queue_url
            == "https://sqs.us-west-2.amazonaws.com/123456789/prod-queue"
        )
        assert config.sns_topic_arn == "arn:aws:sns:us-west-2:123456789:prod-topic"
        assert config.gemini_api_key == "prod-gemini-key"
        assert config.gemini_model == "gemini-1.5-pro"

    @staticmethod
    @patch.dict(
        os.environ,
        {
            # Local development overrides
            "DYNAMODB_ENDPOINT_URL": "http://localhost:8000",
            "S3_ENDPOINT_URL": "http://localhost:9000",
            "GEMINI_MODEL": "gemini-1.5-flash",
        },
        clear=False,
    )
    def test_local_development_config_overrides() -> None:
        """Test local development configuration with endpoint overrides."""
        # Act
        config = AWSConfig()

        # Assert local development settings
        assert config.dynamodb_endpoint_url == "http://localhost:8000"
        assert config.s3_endpoint_url == "http://localhost:9000"
        assert config.gemini_model == "gemini-1.5-flash"
        # Defaults should still apply
        assert config.aws_region == "us-east-1"
        assert config.dynamodb_table_name == "clarity-health-data"
        assert config.s3_bucket_name == "clarity-health-uploads"
