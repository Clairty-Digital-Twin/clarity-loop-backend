#!/usr/bin/env python3
"""Test Cognito authentication directly using boto3."""

import json
import os
import sys

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_cognito_auth() -> None:
    """Test Cognito authentication with minimal setup."""
    # Get configuration from environment
    region = os.getenv("COGNITO_REGION", "us-east-2")
    user_pool_id = os.getenv("COGNITO_USER_POOL_ID", "us-east-2_xqTJHGxmY")
    client_id = os.getenv("COGNITO_CLIENT_ID", "6s5j0f1aiqddqsutrgvg6mjkfr")

    print("Testing Cognito authentication...")
    print(f"Region: {region}")
    print(f"User Pool ID: {user_pool_id}")
    print(f"Client ID: {client_id}")
    print("-" * 50)

    # Create Cognito client
    cognito_client = boto3.client("cognito-idp", region_name=region)

    # First, let's check the app client configuration
    try:
        print("Checking app client configuration...")
        client_info = cognito_client.describe_user_pool_client(
            UserPoolId=user_pool_id, ClientId=client_id
        )

        app_client = client_info.get("UserPoolClient", {})
        print(f"Client Name: {app_client.get('ClientName')}")
        print(f"Auth Flows: {app_client.get('ExplicitAuthFlows', [])}")
        print(f"Has Secret: {'ClientSecret' in app_client}")

        if "ClientSecret" in app_client:
            print(
                "WARNING: App client has a secret configured but backend is not using it!"
            )

        if "ALLOW_USER_PASSWORD_AUTH" not in app_client.get("ExplicitAuthFlows", []):
            print("ERROR: USER_PASSWORD_AUTH flow is not enabled for this app client!")
            print("Available flows:", app_client.get("ExplicitAuthFlows", []))
            return

    except ClientError as e:
        print(f"Error checking app client: {e}")
        return

    print("-" * 50)

    # Test authentication with a test user
    test_email = input("Enter test email (or press Enter to skip auth test): ").strip()
    if not test_email:
        print("Skipping authentication test")
        return

    test_password = input("Enter test password: ").strip()

    try:
        print(f"\nTesting authentication for: {test_email}")

        # Try the exact same call the backend makes
        response = cognito_client.initiate_auth(
            ClientId=client_id,
            AuthFlow="USER_PASSWORD_AUTH",
            AuthParameters={"USERNAME": test_email, "PASSWORD": test_password},
        )

        print("SUCCESS! Authentication worked.")
        print(f"Response keys: {list(response.keys())}")

        if "AuthenticationResult" in response:
            print("Got tokens successfully!")

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_message = e.response["Error"]["Message"]
        print(f"\nERROR: {error_code}")
        print(f"Message: {error_message}")

        # Check for specific errors
        if error_code == "InvalidParameterException":
            print("\nThis error suggests:")
            print("1. The app client might require a SECRET_HASH")
            print("2. Or there's a parameter format issue")

        elif error_code == "NotAuthorizedException":
            print("\nThis error suggests:")
            print("1. Invalid credentials")
            print("2. User not confirmed")
            print("3. User disabled")

        elif error_code == "UserNotFoundException":
            print("\nUser not found in Cognito")

        print("\nFull error response:")
        print(json.dumps(e.response, indent=2))


if __name__ == "__main__":
    test_cognito_auth()
