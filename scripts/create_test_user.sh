#!/usr/bin/env bash
# scripts/create_test_user.sh
set -euo pipefail

EMAIL="${1:-test@example.com}"
PASSWORD="${2:-TestPassword123!}"

echo "Creating test user: $EMAIL"

python3 - "$EMAIL" "$PASSWORD" <<'EOF'
import os
import sys
import boto3
from botocore.exceptions import ClientError

email = sys.argv[1]
password = sys.argv[2]

# Get configuration from environment
region = os.getenv("AWS_REGION", "us-east-1")
pool_id = os.getenv("COGNITO_USER_POOL_ID", "us-east-1_efXaR5EcP")
client_id = os.getenv("COGNITO_CLIENT_ID", "7sm7ckrkovg78b03n1595euc71")

print(f"Using Cognito pool: {pool_id} in region: {region}")

# Create Cognito client
client = boto3.client("cognito-idp", region_name=region)

try:
    # Check if user already exists
    client.admin_get_user(UserPoolId=pool_id, Username=email)
    print(f"✅ User {email} already exists")
except client.exceptions.UserNotFoundException:
    try:
        # Create user
        print(f"Creating user {email}...")
        client.admin_create_user(
            UserPoolId=pool_id,
            Username=email,
            TemporaryPassword=password,
            UserAttributes=[
                {"Name": "email", "Value": email},
                {"Name": "email_verified", "Value": "true"}
            ],
            MessageAction='SUPPRESS'  # Don't send welcome email
        )
        
        # Set permanent password
        client.admin_set_user_password(
            UserPoolId=pool_id,
            Username=email,
            Password=password,
            Permanent=True
        )
        
        print(f"✅ Created test user {email}")
    except Exception as e:
        print(f"❌ Failed to create user: {e}")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error checking user: {e}")
    sys.exit(1)
EOF