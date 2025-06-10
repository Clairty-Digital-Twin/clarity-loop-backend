#!/bin/bash
# Set up AWS Cognito User Pool

set -e

echo "🔐 Setting up AWS Cognito User Pool..."

# Create user pool
USER_POOL_ID=$(aws cognito-idp create-user-pool \
  --pool-name clarity-users \
  --policies '{
    "PasswordPolicy": {
      "MinimumLength": 8,
      "RequireUppercase": true,
      "RequireLowercase": true,
      "RequireNumbers": true,
      "RequireSymbols": false
    }
  }' \
  --auto-verified-attributes email \
  --username-attributes email \
  --mfa-configuration OFF \
  --user-attribute-update-settings '{
    "AttributesRequireVerificationBeforeUpdate": ["email"]
  }' \
  --schema '[
    {
      "Name": "email",
      "AttributeDataType": "String",
      "Required": true,
      "Mutable": true
    },
    {
      "Name": "name",
      "AttributeDataType": "String",
      "Required": false,
      "Mutable": true
    }
  ]' \
  --region us-east-1 \
  --query 'UserPool.Id' \
  --output text)

echo "✅ User pool created: $USER_POOL_ID"

# Create app client
CLIENT_ID=$(aws cognito-idp create-user-pool-client \
  --user-pool-id $USER_POOL_ID \
  --client-name clarity-backend \
  --no-generate-secret \
  --explicit-auth-flows ALLOW_USER_PASSWORD_AUTH ALLOW_REFRESH_TOKEN_AUTH \
  --prevent-user-existence-errors ENABLED \
  --region us-east-1 \
  --query 'UserPoolClient.ClientId' \
  --output text)

echo "✅ App client created: $CLIENT_ID"

# Create a test user (for development)
if [ "$1" == "--create-test-user" ]; then
  echo "Creating test user..."
  aws cognito-idp admin-create-user \
    --user-pool-id $USER_POOL_ID \
    --username test@clarity.com \
    --user-attributes Name=email,Value=test@clarity.com Name=name,Value="Test User" \
    --temporary-password "TempPass123!" \
    --message-action SUPPRESS \
    --region us-east-1
  
  # Set permanent password
  aws cognito-idp admin-set-user-password \
    --user-pool-id $USER_POOL_ID \
    --username test@clarity.com \
    --password "TestPass123!" \
    --permanent \
    --region us-east-1
  
  echo "✅ Test user created: test@clarity.com / TestPass123!"
fi

# Update secrets
aws secretsmanager create-secret \
  --name clarity/cognito-config \
  --secret-string "{\"USER_POOL_ID\":\"$USER_POOL_ID\",\"CLIENT_ID\":\"$CLIENT_ID\"}" \
  --region us-east-1 2>/dev/null || \
aws secretsmanager update-secret \
  --secret-id clarity/cognito-config \
  --secret-string "{\"USER_POOL_ID\":\"$USER_POOL_ID\",\"CLIENT_ID\":\"$CLIENT_ID\"}" \
  --region us-east-1

echo ""
echo "🎉 Cognito setup complete!"
echo "User Pool ID: $USER_POOL_ID"
echo "Client ID: $CLIENT_ID"
echo ""
echo "Add these to your environment:"
echo "export COGNITO_USER_POOL_ID=$USER_POOL_ID"
echo "export COGNITO_CLIENT_ID=$CLIENT_ID"