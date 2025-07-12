#!/bin/bash
# Add Developer Access to ML Models - Secure Implementation
# This script implements proper IAM-based access instead of bucket policy user IDs

set -euo pipefail

BUCKET_NAME="clarity-ml-models-124355672559"
DEVELOPER_GROUP="clarity-developers"
DEVELOPER_POLICY="ClarityDeveloperS3Access"
MATT_USER_ID="AIDARZ5BM7HXWUO5B6GK6"

echo "🔐 Implementing secure developer access to ML models bucket..."
echo "This replaces the insecure user ID-based bucket policy approach"

# Step 1: Create Developer Group if it doesn't exist
echo "📋 Creating developer group..."
aws iam create-group --group-name "$DEVELOPER_GROUP" 2>/dev/null || echo "Group already exists"

# Step 2: Create Developer Policy for S3 ML Models Access
echo "📝 Creating developer policy..."
cat > /tmp/developer-s3-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "MLModelsReadAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": [
        "arn:aws:s3:::clarity-ml-models-124355672559",
        "arn:aws:s3:::clarity-ml-models-124355672559/*"
      ]
    },
    {
      "Sid": "SecureTransportOnly",
      "Effect": "Deny",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::clarity-ml-models-124355672559",
        "arn:aws:s3:::clarity-ml-models-124355672559/*"
      ],
      "Condition": {
        "Bool": {
          "aws:SecureTransport": "false"
        }
      }
    }
  ]
}
EOF

# Step 3: Apply Policy to Group
echo "🔧 Applying policy to developer group..."
aws iam put-group-policy \
    --group-name "$DEVELOPER_GROUP" \
    --policy-name "$DEVELOPER_POLICY" \
    --policy-document file:///tmp/developer-s3-policy.json

# Step 4: Add Matt's User to Developer Group
echo "👤 Adding Matt's user to developer group..."
# Note: We need to find Matt's username from his user ID
MATT_USERNAME=$(aws iam get-user --user-name matt-cofounder --query 'User.UserName' --output text 2>/dev/null || echo "")

if [[ -z "$MATT_USERNAME" ]]; then
    echo "⚠️  Matt's user not found. Please create user first or update script with correct username."
    echo "    User ID: $MATT_USER_ID"
    echo "    To create user: aws iam create-user --user-name matt-cofounder"
    exit 1
fi

aws iam add-user-to-group \
    --group-name "$DEVELOPER_GROUP" \
    --user-name "$MATT_USERNAME"

# Step 5: Update Bucket Policy to Remove User ID Dependencies
echo "🔄 Updating bucket policy to remove user ID dependencies..."
cat > /tmp/secure-bucket-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyInsecureConnections",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::clarity-ml-models-124355672559",
        "arn:aws:s3:::clarity-ml-models-124355672559/*"
      ],
      "Condition": {
        "Bool": {
          "aws:SecureTransport": "false"
        }
      }
    },
    {
      "Sid": "AllowECSTaskRoleAccess",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::124355672559:role/clarity-backend-task-role"
      },
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::clarity-ml-models-124355672559",
        "arn:aws:s3:::clarity-ml-models-124355672559/*"
      ]
    }
  ]
}
EOF

aws s3api put-bucket-policy \
    --bucket "$BUCKET_NAME" \
    --policy file:///tmp/secure-bucket-policy.json

echo "✅ Secure developer access implemented successfully!"
echo ""
echo "🎯 Next Steps:"
echo "1. Matt can now access the bucket using his IAM user credentials"
echo "2. Future developers can be added to the '$DEVELOPER_GROUP' group"
echo "3. No more bucket policy updates needed for new developers"
echo ""
echo "🔍 To verify access:"
echo "aws s3 ls s3://$BUCKET_NAME/ --profile matt-profile"

# Cleanup
rm -f /tmp/developer-s3-policy.json /tmp/secure-bucket-policy.json 