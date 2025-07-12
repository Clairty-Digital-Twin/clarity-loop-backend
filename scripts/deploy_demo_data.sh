#!/bin/bash
# Demo Data Deployment Script for Clarity Digital Twin
# To be run by Matt with PowerUser access

set -euo pipefail

BUCKET_NAME="clarity-demo-data"
REGION="us-east-1"
ACCOUNT_ID="124355672559"

echo "🚀 Clarity Digital Twin Demo Data Deployment"
echo "=============================================="
echo "This script will:"
echo "1. Create S3 bucket for demo data"
echo "2. Upload all demo scenarios"
echo "3. Configure bucket permissions"
echo "4. Validate deployment"
echo ""

# Check AWS credentials
echo "🔐 Checking AWS credentials..."
aws sts get-caller-identity || {
    echo "❌ AWS credentials not configured. Please run 'aws configure'"
    exit 1
}

USER_ARN=$(aws sts get-caller-identity --query Arn --output text)
echo "✅ Authenticated as: $USER_ARN"

# Step 1: Create S3 bucket
echo ""
echo "📦 Creating S3 bucket: $BUCKET_NAME"
if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
    echo "✅ Bucket already exists"
else
    echo "Creating bucket in region: $REGION"
    aws s3api create-bucket \
        --bucket "$BUCKET_NAME" \
        --region "$REGION" || {
        echo "❌ Failed to create bucket. Trying without location constraint..."
        aws s3api create-bucket --bucket "$BUCKET_NAME"
    }
    echo "✅ Bucket created successfully"
fi

# Step 2: Upload demo data
echo ""
echo "📤 Uploading demo data..."
if [ ! -d "demo_data" ]; then
    echo "❌ demo_data directory not found. Please run this script from the project root."
    exit 1
fi

# Sync all demo data
aws s3 sync demo_data/ "s3://$BUCKET_NAME/" --delete
echo "✅ Demo data uploaded successfully"

# Step 3: Configure bucket policy
echo ""
echo "🔒 Configuring bucket permissions..."
cat > /tmp/demo-bucket-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowPublicRead",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::$BUCKET_NAME/*"
    },
    {
      "Sid": "AllowMattFullAccess",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::$ACCOUNT_ID:user/matt-gorbett-cofounder"
      },
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::$BUCKET_NAME",
        "arn:aws:s3:::$BUCKET_NAME/*"
      ]
    },
    {
      "Sid": "AllowECSAccess",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::$ACCOUNT_ID:role/clarity-backend-task-role"
      },
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::$BUCKET_NAME",
        "arn:aws:s3:::$BUCKET_NAME/*"
      ]
    }
  ]
}
EOF

aws s3api put-bucket-policy \
    --bucket "$BUCKET_NAME" \
    --policy file:///tmp/demo-bucket-policy.json
echo "✅ Bucket policy configured"

# Step 4: Enable versioning and lifecycle
echo ""
echo "⚙️  Configuring bucket settings..."
aws s3api put-bucket-versioning \
    --bucket "$BUCKET_NAME" \
    --versioning-configuration Status=Enabled
echo "✅ Versioning enabled"

# Step 5: Validate deployment
echo ""
echo "✅ Validating deployment..."
file_count=$(aws s3 ls "s3://$BUCKET_NAME/" --recursive | wc -l)
echo "📊 Files uploaded: $file_count"

# List key demo files
echo ""
echo "🎯 Key demo files available:"
aws s3 ls "s3://$BUCKET_NAME/" --recursive | grep -E "\.(json|md)$" | head -10

# Generate presigned URLs for key demo files
echo ""
echo "🔗 Demo access URLs (valid for 24 hours):"
aws s3 presign "s3://$BUCKET_NAME/README.md" --expires-in 86400
aws s3 presign "s3://$BUCKET_NAME/healthkit/users.json" --expires-in 86400
aws s3 presign "s3://$BUCKET_NAME/clinical/complete_episode_scenario.json" --expires-in 86400

# Clean up
rm -f /tmp/demo-bucket-policy.json

echo ""
echo "🎉 Demo deployment complete!"
echo ""
echo "📍 Demo data location: s3://$BUCKET_NAME/"
echo "🌐 Public access: https://$BUCKET_NAME.s3.amazonaws.com/"
echo ""
echo "🎬 Demo scenarios ready:"
echo "  • HealthKit Chat: healthkit/ directory"
echo "  • Bipolar Risk Detection: clinical/ directory" 
echo "  • Chat Context: chat/ directory"
echo ""
echo "🚀 Ready to showcase Clarity Digital Twin!" 