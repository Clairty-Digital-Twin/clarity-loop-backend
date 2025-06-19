"""AWS-specific configuration for CLARITY backend."""

# removed - breaks FastAPI

from pydantic import Field
from pydantic_settings import BaseSettings


class AWSConfig(BaseSettings):
    """AWS service configuration."""

    # AWS Core Settings
    aws_region: str = Field(default="us-east-1", description="AWS region")
    aws_access_key_id: str | None = Field(default=None, description="AWS access key ID")
    aws_secret_access_key: str | None = Field(
        default=None, description="AWS secret access key"
    )

    # Cognito Settings
    cognito_user_pool_id: str | None = Field(
        default=None, description="Cognito user pool ID"
    )
    cognito_client_id: str | None = Field(
        default=None, description="Cognito app client ID"
    )
    cognito_region: str | None = Field(default=None, description="Cognito region")

    # DynamoDB Settings
    dynamodb_table_name: str = Field(
        default="clarity-health-data", description="DynamoDB table name"
    )
    dynamodb_endpoint_url: str | None = Field(
        default=None, description="DynamoDB endpoint (for local testing)"
    )

    # S3 Settings
    s3_bucket_name: str = Field(
        default="clarity-health-uploads", description="S3 bucket for file uploads"
    )
    s3_endpoint_url: str | None = Field(
        default=None, description="S3 endpoint (for local testing)"
    )

    # SQS/SNS Settings
    sqs_queue_url: str | None = Field(
        default=None, description="SQS queue URL for async processing"
    )
    sns_topic_arn: str | None = Field(
        default=None, description="SNS topic ARN for notifications"
    )

    # Gemini Settings (keeping this)
    gemini_api_key: str | None = Field(
        default=None, description="Google Gemini API key"
    )
    gemini_model: str = Field(
        default="gemini-1.5-flash", description="Gemini model to use"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
