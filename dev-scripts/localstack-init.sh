#!/bin/bash
# 🏗️ LocalStack Initialization Script
# Sets up all AWS services for local development

set -e

echo "🚀 Initializing LocalStack services..."

# Configuration
AWS_REGION="us-east-1"
AWS_ENDPOINT="http://localhost:4566"
S3_BUCKET="clarity-dev-bucket"
DYNAMODB_TABLE="clarity-dev-health-data"
COGNITO_USER_POOL="clarity-dev-pool"
SQS_QUEUE="clarity-dev-queue"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Wait for LocalStack to be ready
echo -e "${BLUE}⏳ Waiting for LocalStack to be ready...${NC}"
while ! curl -s "$AWS_ENDPOINT/_localstack/health" > /dev/null; do
    sleep 2
done

echo -e "${GREEN}✅ LocalStack is ready!${NC}"

# Create S3 bucket
echo -e "${BLUE}🪣 Creating S3 bucket: $S3_BUCKET${NC}"
aws --endpoint-url="$AWS_ENDPOINT" s3 mb "s3://$S3_BUCKET" || true

# Set up bucket policy for development
aws --endpoint-url="$AWS_ENDPOINT" s3api put-bucket-policy \
    --bucket "$S3_BUCKET" \
    --policy '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": "*",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                "Resource": "arn:aws:s3:::'"$S3_BUCKET"'/*"
            }
        ]
    }' || true

# Create DynamoDB table
echo -e "${BLUE}🗄️  Creating DynamoDB table: $DYNAMODB_TABLE${NC}"
aws --endpoint-url="$AWS_ENDPOINT" dynamodb create-table \
    --table-name "$DYNAMODB_TABLE" \
    --attribute-definitions \
        AttributeName=user_id,AttributeType=S \
        AttributeName=timestamp,AttributeType=S \
    --key-schema \
        AttributeName=user_id,KeyType=HASH \
        AttributeName=timestamp,KeyType=RANGE \
    --provisioned-throughput \
        ReadCapacityUnits=5,WriteCapacityUnits=5 \
    --region "$AWS_REGION" || true

# Create users table
aws --endpoint-url="$AWS_ENDPOINT" dynamodb create-table \
    --table-name "clarity-dev-users" \
    --attribute-definitions \
        AttributeName=user_id,AttributeType=S \
    --key-schema \
        AttributeName=user_id,KeyType=HASH \
    --provisioned-throughput \
        ReadCapacityUnits=5,WriteCapacityUnits=5 \
    --region "$AWS_REGION" || true

# Create Cognito User Pool
echo -e "${BLUE}👥 Creating Cognito User Pool: $COGNITO_USER_POOL${NC}"
USER_POOL_ID=$(aws --endpoint-url="$AWS_ENDPOINT" cognito-idp create-user-pool \
    --pool-name "$COGNITO_USER_POOL" \
    --policies '{
        "PasswordPolicy": {
            "MinimumLength": 8,
            "RequireUppercase": false,
            "RequireLowercase": false,
            "RequireNumbers": false,
            "RequireSymbols": false
        }
    }' \
    --auto-verified-attributes email \
    --region "$AWS_REGION" \
    --query 'UserPool.Id' \
    --output text 2>/dev/null || echo "us-east-1_DevPool123")

echo -e "${YELLOW}📝 User Pool ID: $USER_POOL_ID${NC}"

# Create Cognito User Pool Client
CLIENT_ID=$(aws --endpoint-url="$AWS_ENDPOINT" cognito-idp create-user-pool-client \
    --user-pool-id "$USER_POOL_ID" \
    --client-name "clarity-dev-client" \
    --no-generate-secret \
    --explicit-auth-flows ADMIN_NO_SRP_AUTH ALLOW_USER_PASSWORD_AUTH ALLOW_REFRESH_TOKEN_AUTH \
    --region "$AWS_REGION" \
    --query 'UserPoolClient.ClientId' \
    --output text 2>/dev/null || echo "dev-client-123")

echo -e "${YELLOW}📝 Client ID: $CLIENT_ID${NC}"

# Create test users
echo -e "${BLUE}👤 Creating test users...${NC}"

# Create admin user
aws --endpoint-url="$AWS_ENDPOINT" cognito-idp admin-create-user \
    --user-pool-id "$USER_POOL_ID" \
    --username "admin@clarity.dev" \
    --user-attributes \
        Name=email,Value="admin@clarity.dev" \
        Name=email_verified,Value="true" \
        Name=given_name,Value="Admin" \
        Name=family_name,Value="User" \
    --temporary-password "TempPass123!" \
    --message-action SUPPRESS \
    --region "$AWS_REGION" 2>/dev/null || true

# Set permanent password for admin
aws --endpoint-url="$AWS_ENDPOINT" cognito-idp admin-set-user-password \
    --user-pool-id "$USER_POOL_ID" \
    --username "admin@clarity.dev" \
    --password "DevPass123!" \
    --permanent \
    --region "$AWS_REGION" 2>/dev/null || true

# Create regular test user
aws --endpoint-url="$AWS_ENDPOINT" cognito-idp admin-create-user \
    --user-pool-id "$USER_POOL_ID" \
    --username "testuser@clarity.dev" \
    --user-attributes \
        Name=email,Value="testuser@clarity.dev" \
        Name=email_verified,Value="true" \
        Name=given_name,Value="Test" \
        Name=family_name,Value="User" \
    --temporary-password "TempPass123!" \
    --message-action SUPPRESS \
    --region "$AWS_REGION" 2>/dev/null || true

# Set permanent password for test user
aws --endpoint-url="$AWS_ENDPOINT" cognito-idp admin-set-user-password \
    --user-pool-id "$USER_POOL_ID" \
    --username "testuser@clarity.dev" \
    --password "DevPass123!" \
    --permanent \
    --region "$AWS_REGION" 2>/dev/null || true

# Create SQS queue
echo -e "${BLUE}📬 Creating SQS queue: $SQS_QUEUE${NC}"
aws --endpoint-url="$AWS_ENDPOINT" sqs create-queue \
    --queue-name "$SQS_QUEUE" \
    --region "$AWS_REGION" || true

# Create SNS topic
echo -e "${BLUE}📢 Creating SNS topic...${NC}"
aws --endpoint-url="$AWS_ENDPOINT" sns create-topic \
    --name "clarity-dev-notifications" \
    --region "$AWS_REGION" || true

# Upload sample ML models to S3
echo -e "${BLUE}🧠 Uploading sample ML models...${NC}"
mkdir -p /tmp/sample-models

# Create dummy model files
echo "# Dummy PAT model for development" > /tmp/sample-models/pat-model.txt
echo "# Dummy model weights" > /tmp/sample-models/model-weights.bin

aws --endpoint-url="$AWS_ENDPOINT" s3 cp /tmp/sample-models/pat-model.txt "s3://$S3_BUCKET/models/pat/" || true
aws --endpoint-url="$AWS_ENDPOINT" s3 cp /tmp/sample-models/model-weights.bin "s3://$S3_BUCKET/models/pat/" || true

# Seed some sample data
echo -e "${BLUE}🌱 Seeding sample health data...${NC}"

# Create sample health data JSON
cat > /tmp/sample-health-data.json << EOF
{
    "user_id": "testuser@clarity.dev",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "heart_rate": 72,
    "steps": 8500,
    "sleep_hours": 7.5,
    "activity_type": "walking",
    "data_source": "development_seed"
}
EOF

# Upload to DynamoDB
aws --endpoint-url="$AWS_ENDPOINT" dynamodb put-item \
    --table-name "$DYNAMODB_TABLE" \
    --item file:///tmp/sample-health-data.json \
    --region "$AWS_REGION" || true

# Clean up temp files
rm -rf /tmp/sample-models /tmp/sample-health-data.json

echo -e "${GREEN}🎉 LocalStack initialization complete!${NC}"
echo ""
echo -e "${YELLOW}📋 Development Resources Created:${NC}"
echo -e "  🪣 S3 Bucket: $S3_BUCKET"
echo -e "  🗄️  DynamoDB Tables: $DYNAMODB_TABLE, clarity-dev-users"
echo -e "  👥 Cognito User Pool: $USER_POOL_ID"
echo -e "  📱 Client ID: $CLIENT_ID"
echo -e "  📬 SQS Queue: $SQS_QUEUE"
echo -e "  👤 Test Users:"
echo -e "     • admin@clarity.dev (password: DevPass123!)"
echo -e "     • testuser@clarity.dev (password: DevPass123!)"
echo ""
echo -e "${GREEN}🚀 Ready for development!${NC}"