#!/bin/bash
# Add co-founder (Matt) to AWS with proper IAM access
# This gives him permanent but LIMITED access to only what he needs

set -e

COFOUNDER_EMAIL="${1:-matthewgorbett@gmail.com}"
COFOUNDER_NAME="${2:-Matt-Gorbett}"
BUCKET_NAME="clarity-ml-models-124355672559"

echo "🤝 Setting up AWS access for co-founder: ${COFOUNDER_NAME}"
echo ""

# Create IAM policy for model access
cat > /tmp/cofounder-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "ReadPATModels",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::${BUCKET_NAME}",
                "arn:aws:s3:::${BUCKET_NAME}/pat/*"
            ]
        },
        {
            "Sid": "ListBuckets",
            "Effect": "Allow",
            "Action": "s3:ListAllMyBuckets",
            "Resource": "*"
        }
    ]
}
EOF

# Create the IAM policy
echo "📋 Creating IAM policy..."
POLICY_ARN=$(aws iam create-policy \
    --policy-name "ClarityCofounderModelAccess" \
    --policy-document file:///tmp/cofounder-policy.json \
    --description "Allows co-founders to download ML models" \
    --query 'Policy.Arn' \
    --output text 2>/dev/null || \
    aws iam list-policies --query "Policies[?PolicyName=='ClarityCofounderModelAccess'].Arn | [0]" --output text)

# Create IAM user
echo "👤 Creating IAM user..."
aws iam create-user --user-name "${COFOUNDER_NAME}" 2>/dev/null || echo "User already exists"

# Attach policy to user
echo "🔗 Attaching policy..."
aws iam attach-user-policy \
    --user-name "${COFOUNDER_NAME}" \
    --policy-arn "${POLICY_ARN}"

# Create access key
echo "🔑 Creating access credentials..."
CREDENTIALS=$(aws iam create-access-key --user-name "${COFOUNDER_NAME}" --query 'AccessKey.[AccessKeyId,SecretAccessKey]' --output text)
ACCESS_KEY=$(echo $CREDENTIALS | cut -d' ' -f1)
SECRET_KEY=$(echo $CREDENTIALS | cut -d' ' -f2)

# Create a credentials file for Matt
cat > /tmp/matt-aws-credentials.txt << EOF
🎉 Welcome to the team, Matt! 🎉

Here are your AWS credentials for accessing Clarity ML models:

[default]
aws_access_key_id = ${ACCESS_KEY}
aws_secret_access_key = ${SECRET_KEY}
region = us-east-1

SETUP INSTRUCTIONS:
1. Create ~/.aws/credentials file on your machine
2. Copy the above [default] section into it
3. The download_models.sh script will now work!

ALTERNATIVE: Set environment variables:
export AWS_ACCESS_KEY_ID="${ACCESS_KEY}"
export AWS_SECRET_ACCESS_KEY="${SECRET_KEY}"
export AWS_DEFAULT_REGION="us-east-1"

Your access is limited to:
✅ Downloading PAT model weights
✅ Listing model files
❌ No access to other AWS resources (secure!)

Welcome to Clarity! 🚀
EOF

echo ""
echo "✅ Co-founder access created successfully!"
echo ""
echo "📧 IMPORTANT: Send the file /tmp/matt-aws-credentials.txt to Matt SECURELY"
echo "   (Use encrypted email, Signal, or another secure channel)"
echo ""
echo "🔒 Security notes:"
echo "   - Matt only has READ access to ML models"
echo "   - No access to other AWS resources"
echo "   - You can revoke access anytime with:"
echo "     aws iam delete-access-key --access-key-id ${ACCESS_KEY} --user-name ${COFOUNDER_NAME}"

# Clean up policy file
rm -f /tmp/cofounder-policy.json