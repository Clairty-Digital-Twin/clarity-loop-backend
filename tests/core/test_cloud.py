"""Test cloud module."""

from unittest.mock import Mock, patch
import pytest
from clarity.core.cloud import (
    get_aws_session,
    get_cognito_client,
    get_dynamodb_resource,
    gemini_api_key,
)


def test_get_aws_session():
    """Test get_aws_session returns a boto3 session."""
    with patch("clarity.core.cloud.boto3.Session") as mock_session:
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        
        result = get_aws_session()
        
        assert result == mock_session_instance
        mock_session.assert_called_once_with(region_name="us-east-1")


def test_get_aws_session_custom_region():
    """Test get_aws_session with custom region."""
    with patch("clarity.core.cloud.boto3.Session") as mock_session:
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        
        result = get_aws_session(region_name="eu-west-1")
        
        assert result == mock_session_instance
        mock_session.assert_called_once_with(region_name="eu-west-1")


def test_get_cognito_client():
    """Test get_cognito_client returns a Cognito client."""
    with patch("clarity.core.cloud.boto3.client") as mock_client:
        mock_cognito_client = Mock()
        mock_client.return_value = mock_cognito_client
        
        result = get_cognito_client()
        
        assert result == mock_cognito_client
        mock_client.assert_called_once_with("cognito-idp", region_name="us-east-1")


def test_get_dynamodb_resource():
    """Test get_dynamodb_resource returns a DynamoDB resource."""
    with patch("clarity.core.cloud.boto3.resource") as mock_resource:
        mock_dynamodb_resource = Mock()
        mock_resource.return_value = mock_dynamodb_resource
        
        result = get_dynamodb_resource()
        
        assert result == mock_dynamodb_resource
        mock_resource.assert_called_once_with("dynamodb", region_name="us-east-1")


def test_gemini_api_key_exists():
    """Test gemini_api_key when environment variable is set."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key-123"}):
        result = gemini_api_key()
        assert result == "test-key-123"


def test_gemini_api_key_not_exists():
    """Test gemini_api_key when environment variable is not set."""
    with patch.dict("os.environ", {}, clear=True):
        result = gemini_api_key()
        assert result is None