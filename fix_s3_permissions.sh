#!/bin/bash

# Fix S3 permissions for PAT model downloads
# This script attaches the S3 policy to the correct ECS task role

set -e

ROLE_NAME="clarity-backend-task-role"
POLICY_NAME="ClarityECSTaskS3Policy"
POLICY_FILE="ops/iam/ecs-task-s3-policy.json"

echo "🔧 Fixing S3 permissions for PAT model downloads..."

# Check if role exists
if ! aws iam get-role --role-name "$ROLE_NAME" &>/dev/null; then
    echo "❌ Role $ROLE_NAME does not exist!"
    exit 1
fi

# Check if policy file exists
if [ ! -f "$POLICY_FILE" ]; then
    echo "❌ Policy file $POLICY_FILE does not exist!"
    exit 1
fi

echo "✅ Role exists: $ROLE_NAME"
echo "✅ Policy file exists: $POLICY_FILE"

# Create/update managed policy
echo "📝 Creating/updating managed policy: $POLICY_NAME"
POLICY_ARN="arn:aws:iam::124355672559:policy/$POLICY_NAME"

# Try to create the policy (will fail if it already exists)
if aws iam create-policy \
    --policy-name "$POLICY_NAME" \
    --policy-document file://"$POLICY_FILE" \
    --description "Allow ECS tasks to access S3 bucket for ML models" 2>/dev/null; then
    echo "✅ Policy created successfully"
else
    echo "📝 Policy already exists, updating..."
    # Update the policy version
    aws iam create-policy-version \
        --policy-arn "$POLICY_ARN" \
        --policy-document file://"$POLICY_FILE" \
        --set-as-default
    echo "✅ Policy updated successfully"
fi

# Attach policy to role
echo "🔗 Attaching policy to role..."
if aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn "$POLICY_ARN" 2>/dev/null; then
    echo "✅ Policy attached successfully"
else
    echo "✅ Policy was already attached"
fi

# Verify the attachment
echo "🔍 Verifying policy attachment..."
if aws iam list-attached-role-policies --role-name "$ROLE_NAME" --query "AttachedPolicies[?PolicyName=='$POLICY_NAME']" --output text | grep -q "$POLICY_NAME"; then
    echo "✅ Policy is attached to role"
else
    echo "❌ Policy attachment verification failed"
    exit 1
fi

echo ""
echo "🎉 SUCCESS! S3 permissions have been fixed!"
echo "🔧 The ECS task role now has permissions to:"
echo "   - s3:ListBucket on clarity-ml-models-124355672559"
echo "   - s3:GetObject on clarity-ml-models-124355672559/*"
echo ""
echo "🚀 The PAT model download should now work in production!"
echo "💡 You may need to restart the ECS service for the changes to take effect." 