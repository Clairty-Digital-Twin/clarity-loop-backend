#!/usr/bin/env python3
"""Create a demo user in AWS Cognito for testing the live ML pipeline.

This bypasses shell integration issues by using boto3 directly.
"""

import json

import boto3
from botocore.exceptions import ClientError
import requests

# Configuration from ECS task definition
USER_POOL_ID = "us-east-1_efXaR5EcP"
CLIENT_ID = "7sm7ckrkovg78b03n1595euc71"
REGION = "us-east-1"


def create_demo_user() -> bool | None:
    """Create a demo user in Cognito user pool."""
    cognito = boto3.client('cognito-idp', region_name=REGION)

    username = "demo@clarity.ai"
    temp_password = "TempDemo123!"  # noqa: S105

    try:
        # Create user
        cognito.admin_create_user(
            UserPoolId=USER_POOL_ID,
            Username=username,
            UserAttributes=[
                {
                    'Name': 'email',
                    'Value': username
                },
                {
                    'Name': 'email_verified',
                    'Value': 'true'
                }
            ],
            TemporaryPassword=temp_password,
            MessageAction='SUPPRESS'
        )

        print(f"✅ Demo user created: {username}")
        print(f"📧 Email: {username}")
        print(f"🔑 Temporary password: {temp_password}")

        # Set permanent password
        cognito.admin_set_user_password(
            UserPoolId=USER_POOL_ID,
            Username=username,
            Password="DemoPassword123!",  # noqa: S106
            Permanent=True
        )

        print("✅ Permanent password set: DemoPassword123!")

        return True

    except ClientError as e:
        if e.response['Error']['Code'] == 'UsernameExistsException':
            print(f"✅ User {username} already exists!")
            return True
        print(f"❌ Error creating user: {e}")
        return False


def get_jwt_token_via_api() -> str | None:
    """Get JWT token using the live API login endpoint."""
    API_BASE = "https://clarity.novamindnyc.com/api/v1"

    login_data = {
        "email": "demo@clarity.ai",
        "password": "DemoPassword123!"
    }

    try:
        response = requests.post(
            f"{API_BASE}/auth/login",
            json=login_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )

        print(f"🔑 Login Response: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            token = result.get('access_token')
            if token:
                print("🎉 JWT Token obtained via API!")
                print(f"🔑 Token: {token[:50]}...")
                return token
            print("❌ No access_token in response")
            return None
        print(f"❌ Login failed: {response.text}")
        return None

    except Exception as e:
        print(f"❌ API login error: {e}")
        return None


def test_live_ml_pipeline(token: str) -> bool:
    """Test the live ML pipeline with real authentication."""
    API_BASE = "https://clarity.novamindnyc.com/api/v1"

    # Test authenticated endpoint with REAL step data
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    # Real Apple HealthKit step data format (based on the API spec)
    step_data = {
        "step_counts": [
            1200, 1450, 980, 1600, 2200, 1800, 1500, 1300, 1100, 950,
            1400, 1700, 1900, 2100, 1650, 1450, 1200, 1000, 800, 600,
            400, 200, 100, 50  # 24 hours of step data
        ],
        "timestamps": [
            "2024-01-15T00:00:00Z", "2024-01-15T01:00:00Z", "2024-01-15T02:00:00Z",
            "2024-01-15T03:00:00Z", "2024-01-15T04:00:00Z", "2024-01-15T05:00:00Z",
            "2024-01-15T06:00:00Z", "2024-01-15T07:00:00Z", "2024-01-15T08:00:00Z",
            "2024-01-15T09:00:00Z", "2024-01-15T10:00:00Z", "2024-01-15T11:00:00Z",
            "2024-01-15T12:00:00Z", "2024-01-15T13:00:00Z", "2024-01-15T14:00:00Z",
            "2024-01-15T15:00:00Z", "2024-01-15T16:00:00Z", "2024-01-15T17:00:00Z",
            "2024-01-15T18:00:00Z", "2024-01-15T19:00:00Z", "2024-01-15T20:00:00Z",
            "2024-01-15T21:00:00Z", "2024-01-15T22:00:00Z", "2024-01-15T23:00:00Z"
        ]
    }

    try:
        print("🚀 Sending real step data to PAT ML pipeline...")

        response = requests.post(
            f"{API_BASE}/pat/step-analysis",
            json=step_data,
            headers=headers,
            timeout=60  # ML processing can take time
        )

        print(f"🎯 ML Pipeline Response: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("🎉 LIVE ML PROCESSING SUCCESS!")
            print("📊 PAT Model Results:")
            print(f"   Analysis ID: {result.get('analysis_id', 'N/A')}")
            print(f"   Status: {result.get('status', 'N/A')}")
            print(f"   Quality Score: {result.get('quality_score', 'N/A')}")
            if 'insights' in result:
                print(f"   Insights: {result['insights']}")
            if 'pat_predictions' in result:
                print(f"   PAT Predictions: {result['pat_predictions']}")
            print(f"📋 Full Result: {json.dumps(result, indent=2)}")
            return True
        if response.status_code == 422:
            print("❌ Validation Error - checking data format...")
            print(f"   Response: {response.text}")
            return False
        print(f"❌ Error: {response.status_code}")
        print(f"   Response: {response.text}")
        return False

    except Exception as e:
        print(f"❌ Request error: {e}")
        return False


def test_multiple_endpoints(token: str) -> None:
    """Test multiple ML endpoints to see what's available."""
    API_BASE = "https://clarity.novamindnyc.com/api/v1"

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    endpoints_to_test = [
        ('/pat/health', 'GET'),
        ('/pat/step-analysis', 'POST'),
        ('/pat/analysis', 'POST'),
        ('/health-data/health', 'GET'),
        ('/insights/generate', 'POST'),
        ('/healthkit/upload', 'POST')
    ]

    print("🔍 Testing multiple endpoints...")

    for endpoint, method in endpoints_to_test:
        try:
            if method == 'GET':
                response = requests.get(
                    f"{API_BASE}{endpoint}",
                    headers=headers,
                    timeout=10
                )
            else:  # POST
                response = requests.post(
                    f"{API_BASE}{endpoint}",
                    json={"test": "data"},
                    headers=headers,
                    timeout=10
                )

            print(f"  {method} {endpoint}: {response.status_code}")

            if response.status_code == 200:
                print("    ✅ Available!")
            elif response.status_code == 405:
                print("    ⚠️ Method not allowed")
            elif response.status_code == 404:
                print("    ❌ Not found")
            elif response.status_code == 422:
                print("    ⚠️ Validation error (needs proper data)")
            else:
                print(f"    ❓ Status: {response.status_code}")

        except Exception as e:
            print(f"    ❌ Error: {e}")


if __name__ == "__main__":
    print("🎯 Creating Cognito Demo User for Live ML Pipeline...")
    print("=" * 60)

    # Step 1: Create user
    if create_demo_user():
        print("\n🔑 Getting JWT Token via API...")

        # Step 2: Get JWT token via API
        token = get_jwt_token_via_api()

        if token:
            print("\n🚀 Testing Live ML Pipeline...")

            # Step 3: Test multiple endpoints
            test_multiple_endpoints(token)

            # Step 4: Test live ML pipeline
            success = test_live_ml_pipeline(token)

            if success:
                print("\n🎉 GOD STATUS ACHIEVED! LIVE ML PIPELINE WORKING!")
                print("✅ Demo user created")
                print("✅ JWT authentication working")
                print("✅ Live ML processing confirmed")
            else:
                print("\n⚠️ Pipeline accessible but needs data format adjustment")
        else:
            print("\n❌ Could not get JWT token")
    else:
        print("\n❌ Could not create demo user")
