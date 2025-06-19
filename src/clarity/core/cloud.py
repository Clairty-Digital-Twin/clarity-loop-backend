"""CLARITY Digital Twin Platform - AWS Cloud Credentials Helper.

This module provides centralized access to AWS credentials and API keys.
"""

# removed - breaks FastAPI

import os
from typing import TYPE_CHECKING

import boto3
from mypy_boto3_cognito_idp import CognitoIdentityProviderClient
from mypy_boto3_dynamodb import DynamoDBServiceResource

if TYPE_CHECKING:
    pass  # Only for type stubs now


def get_aws_session(region_name: str = "us-east-1") -> boto3.Session:
    """Get AWS session with configured credentials.

    Args:
        region_name: AWS region name

    Returns:
        boto3.Session: Configured AWS session
    """
    return boto3.Session(region_name=region_name)


def get_cognito_client(region_name: str = "us-east-1") -> CognitoIdentityProviderClient:
    """Get AWS Cognito client.

    Args:
        region_name: AWS region name

    Returns:
        boto3.client: Cognito Identity Provider client
    """
    return boto3.client("cognito-idp", region_name=region_name)


def get_dynamodb_resource(region_name: str = "us-east-1") -> DynamoDBServiceResource:
    """Get AWS DynamoDB resource.

    Args:
        region_name: AWS region name

    Returns:
        boto3.resource: DynamoDB resource
    """
    return boto3.resource("dynamodb", region_name=region_name)


def gemini_api_key() -> str | None:
    """Get Gemini API key from environment variables.

    Returns:
        str: The Gemini API key, or None if not set
    """
    return os.environ.get("GEMINI_API_KEY")
