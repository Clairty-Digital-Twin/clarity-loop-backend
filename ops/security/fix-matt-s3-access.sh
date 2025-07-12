#!/bin/bash
# Quick fix: Add Matt's user ID to S3 bucket policy

set -euo pipefail

BUCKET_NAME="clarity-ml-models-124355672559"
MATT_USER_ID="AIDARZ5BM7HXWUO5B6GK6"

echo "🔧 Adding Matt's access to S3 bucket..."

# Step 1: Get current bucket policy
echo "📋 Getting current bucket policy..."
aws s3api get-bucket-policy --bucket "$BUCKET_NAME" --query Policy --output text > /tmp/current-policy.json

# Step 2: Update the policy to include Matt's user ID
echo "✏️  Updating policy to include Matt's user ID..."
cat > /tmp/updated-policy.json << 'EOF'
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
    },
    {
      "Sid": "DenyUnauthorizedAccess",
      "Effect": "Deny",
      "Principal": "*",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::clarity-ml-models-124355672559",
        "arn:aws:s3:::clarity-ml-models-124355672559/*"
      ],
      "Condition": {
        "StringNotLike": {
          "aws:userid": [
            "AROARZ5BM7HX7S5M4DLSZ:*",
            "AIDARZ5BM7HXRJRCYFWYV:*",
            "AIDARZ5BM7HXWUO5B6GK6",
            "124355672559"
          ]
        }
      }
    }
  ]
}
EOF

# Step 3: Apply updated policy
echo "🚀 Applying updated bucket policy..."
aws s3api put-bucket-policy --bucket "$BUCKET_NAME" --policy file:///tmp/updated-policy.json

echo "✅ Matt's access added successfully!"
echo "Matt can now access the bucket with his credentials:"
echo "aws s3 ls s3://$BUCKET_NAME/"

# Cleanup
rm -f /tmp/current-policy.json /tmp/updated-policy.json 