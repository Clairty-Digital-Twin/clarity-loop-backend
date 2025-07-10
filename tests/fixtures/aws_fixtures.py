"""AWS service fixtures using moto for isolated testing.

Provides reusable test fixtures for DynamoDB, S3, and Cognito
with proper setup, teardown, and helper utilities.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
import json
import os
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import boto3
from moto import mock_cognito_idp, mock_dynamodb, mock_s3
from mypy_boto3_cognito_idp import CognitoIdentityProviderClient
from mypy_boto3_dynamodb import DynamoDBClient, DynamoDBServiceResource
from mypy_boto3_s3 import S3Client
import pytest

if TYPE_CHECKING:
    from mypy_boto3_dynamodb.service_resource import Table


# Test configuration
TEST_REGION = "us-east-1"
TEST_USER_POOL_NAME = "test-user-pool"
TEST_USER_POOL_CLIENT_NAME = "test-client"
TEST_BUCKET_NAME = "test-clarity-bucket"
TEST_TABLE_NAME = "test-clarity-table"


@pytest.fixture(scope="function")
def aws_credentials():
    """Mock AWS credentials for testing."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = TEST_REGION


@pytest.fixture(scope="function")
def dynamodb_mock(aws_credentials):
    """Create a mocked DynamoDB service."""
    with mock_dynamodb():
        yield boto3.resource("dynamodb", region_name=TEST_REGION)


@pytest.fixture(scope="function")
def dynamodb_client(aws_credentials) -> Generator[DynamoDBClient, None, None]:
    """Create a mocked DynamoDB client."""
    with mock_dynamodb():
        yield boto3.client("dynamodb", region_name=TEST_REGION)


@pytest.fixture(scope="function")
def dynamodb_table(dynamodb_mock: DynamoDBServiceResource) -> Table:
    """Create a test DynamoDB table with common attributes."""
    table = dynamodb_mock.create_table(
        TableName=TEST_TABLE_NAME,
        KeySchema=[
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
            {"AttributeName": "gsi1pk", "AttributeType": "S"},
            {"AttributeName": "gsi1sk", "AttributeType": "S"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "GSI1",
                "KeySchema": [
                    {"AttributeName": "gsi1pk", "KeyType": "HASH"},
                    {"AttributeName": "gsi1sk", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
                "BillingMode": "PAY_PER_REQUEST",
            }
        ],
        BillingMode="PAY_PER_REQUEST",
    )

    # Wait for table to be active
    table.wait_until_exists()
    return table


@pytest.fixture(scope="function")
def s3_mock(aws_credentials) -> Generator[S3Client, None, None]:
    """Create a mocked S3 client."""
    with mock_s3():
        client = boto3.client("s3", region_name=TEST_REGION)
        # Create test bucket
        client.create_bucket(Bucket=TEST_BUCKET_NAME)
        yield client


@pytest.fixture(scope="function")
def s3_bucket(s3_mock: S3Client) -> dict[str, Any]:
    """Create a test S3 bucket with common configuration."""
    # Enable versioning
    s3_mock.put_bucket_versioning(
        Bucket=TEST_BUCKET_NAME,
        VersioningConfiguration={"Status": "Enabled"},
    )

    # Add lifecycle configuration
    s3_mock.put_bucket_lifecycle_configuration(
        Bucket=TEST_BUCKET_NAME,
        LifecycleConfiguration={
            "Rules": [
                {
                    "ID": "DeleteOldVersions",
                    "Status": "Enabled",
                    "NoncurrentVersionExpiration": {"NoncurrentDays": 30},
                },
                {
                    "ID": "TransitionToIA",
                    "Status": "Enabled",
                    "Transitions": [
                        {
                            "Days": 90,
                            "StorageClass": "STANDARD_IA",
                        }
                    ],
                },
            ]
        },
    )

    return {
        "name": TEST_BUCKET_NAME,
        "client": s3_mock,
        "region": TEST_REGION,
    }


@pytest.fixture(scope="function")
def cognito_mock(
    aws_credentials,
) -> Generator[CognitoIdentityProviderClient, None, None]:
    """Create a mocked Cognito Identity Provider client."""
    with mock_cognito_idp():
        yield boto3.client("cognito-idp", region_name=TEST_REGION)


@pytest.fixture(scope="function")
def cognito_user_pool(cognito_mock: CognitoIdentityProviderClient) -> dict[str, str]:
    """Create a test Cognito user pool with client."""
    # Create user pool
    pool_response = cognito_mock.create_user_pool(
        PoolName=TEST_USER_POOL_NAME,
        Policies={
            "PasswordPolicy": {
                "MinimumLength": 8,
                "RequireUppercase": True,
                "RequireLowercase": True,
                "RequireNumbers": True,
                "RequireSymbols": True,
            }
        },
        AutoVerifiedAttributes=["email"],
        Schema=[
            {
                "Name": "email",
                "AttributeDataType": "String",
                "Required": True,
                "Mutable": True,
            },
            {
                "Name": "custom:role",
                "AttributeDataType": "String",
                "Mutable": True,
            },
        ],
    )

    user_pool_id = pool_response["UserPool"]["Id"]

    # Create app client
    client_response = cognito_mock.create_user_pool_client(
        UserPoolId=user_pool_id,
        ClientName=TEST_USER_POOL_CLIENT_NAME,
        GenerateSecret=False,
        ExplicitAuthFlows=[
            "ADMIN_NO_SRP_AUTH",
            "USER_PASSWORD_AUTH",
            "ALLOW_REFRESH_TOKEN_AUTH",
        ],
    )

    return {
        "user_pool_id": user_pool_id,
        "client_id": client_response["UserPoolClient"]["ClientId"],
        "region": TEST_REGION,
    }


@pytest.fixture(scope="function")
def cognito_test_user(
    cognito_mock: CognitoIdentityProviderClient,
    cognito_user_pool: dict[str, str],
) -> dict[str, str]:
    """Create a test user in Cognito."""
    username = "test@example.com"
    password = "TestPassword123!"

    # Create user
    cognito_mock.admin_create_user(
        UserPoolId=cognito_user_pool["user_pool_id"],
        Username=username,
        UserAttributes=[
            {"Name": "email", "Value": username},
            {"Name": "email_verified", "Value": "true"},
            {"Name": "custom:role", "Value": "patient"},
        ],
        TemporaryPassword=password,
        MessageAction="SUPPRESS",
    )

    # Set permanent password
    cognito_mock.admin_set_user_password(
        UserPoolId=cognito_user_pool["user_pool_id"],
        Username=username,
        Password=password,
        Permanent=True,
    )

    return {
        "username": username,
        "password": password,
        "user_pool_id": cognito_user_pool["user_pool_id"],
        "client_id": cognito_user_pool["client_id"],
    }


# Helper fixtures for common test data


@pytest.fixture
def sample_dynamodb_items() -> list[dict[str, Any]]:
    """Sample DynamoDB items for testing."""
    return [
        {
            "pk": "USER#test-user-123",
            "sk": "PROFILE",
            "email": "test@example.com",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "gsi1pk": "EMAIL#test@example.com",
            "gsi1sk": "USER#test-user-123",
        },
        {
            "pk": "USER#test-user-123",
            "sk": "HEALTH_DATA#2024-01-01",
            "metrics": {
                "heart_rate": 72,
                "steps": 8500,
                "sleep_hours": 7.5,
            },
            "created_at": "2024-01-01T00:00:00Z",
        },
        {
            "pk": "USER#test-user-456",
            "sk": "PROFILE",
            "email": "another@example.com",
            "created_at": "2024-01-02T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
            "gsi1pk": "EMAIL#another@example.com",
            "gsi1sk": "USER#test-user-456",
        },
    ]


@pytest.fixture
def sample_s3_objects() -> list[dict[str, Any]]:
    """Sample S3 objects for testing."""
    return [
        {
            "key": "models/pat/v1.0/model.pth",
            "body": b"mock model data",
            "content_type": "application/octet-stream",
            "metadata": {"version": "1.0", "type": "pytorch"},
        },
        {
            "key": "data/users/test-user-123/health_data.json",
            "body": json.dumps(
                {"heart_rate": [72, 74, 71, 73], "timestamps": ["2024-01-01T00:00:00Z"]}
            ),
            "content_type": "application/json",
            "metadata": {"user_id": "test-user-123"},
        },
        {
            "key": "exports/analysis/2024-01-01/report.pdf",
            "body": b"mock pdf report",
            "content_type": "application/pdf",
            "metadata": {"generated_at": "2024-01-01T12:00:00Z"},
        },
    ]


# Batch operation helpers


@pytest.fixture
def dynamodb_batch_writer(dynamodb_table: Table):
    """Helper for batch writing to DynamoDB."""

    def write_items(items: list[dict[str, Any]]) -> None:
        with dynamodb_table.batch_writer() as batch:
            for item in items:
                batch.put_item(Item=item)

    return write_items


@pytest.fixture
def s3_batch_uploader(s3_mock: S3Client):
    """Helper for batch uploading to S3."""

    def upload_objects(objects: list[dict[str, Any]]) -> None:
        for obj in objects:
            s3_mock.put_object(
                Bucket=TEST_BUCKET_NAME,
                Key=obj["key"],
                Body=obj["body"],
                ContentType=obj.get("content_type", "application/octet-stream"),
                Metadata=obj.get("metadata", {}),
            )

    return upload_objects


# Circuit breaker fixture for resilience testing


@pytest.fixture
def mock_circuit_breaker():
    """Mock circuit breaker for testing resilience patterns."""
    from unittest.mock import MagicMock

    breaker = MagicMock()
    breaker.call.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
    breaker.state = "closed"
    breaker.failure_count = 0
    breaker.success_count = 0

    return breaker
