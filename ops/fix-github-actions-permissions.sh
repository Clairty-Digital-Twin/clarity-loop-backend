#!/bin/bash

# Fix GitHub Actions Deploy Role Permissions
# This script adds the missing permissions required for the Security Smoke Test

set -e

ROLE_NAME="GitHubActionsDeploy"
POLICY_NAME="GitHubActionsDeployPolicy"
ACCOUNT_ID="124355672559"
REGION="us-east-1"

echo "🔧 Fixing GitHub Actions Deploy Role Permissions..."
echo "📋 Role: $ROLE_NAME"
echo "📋 Policy: $POLICY_NAME"
echo "📋 Account: $ACCOUNT_ID"
echo "📋 Region: $REGION"
echo ""

# Check if the role exists
if ! aws iam get-role --role-name $ROLE_NAME >/dev/null 2>&1; then
    echo "❌ Role $ROLE_NAME does not exist"
    exit 1
fi

echo "✅ Role $ROLE_NAME exists"

# Check if the policy file exists
if [ ! -f "github-actions-deploy-role-policy.json" ]; then
    echo "❌ Policy file github-actions-deploy-role-policy.json not found"
    echo "Please ensure you're running this script from the ops/ directory"
    exit 1
fi

echo "✅ Policy file found"

# Apply the policy
echo "🔐 Applying IAM policy..."
aws iam put-role-policy \
    --role-name $ROLE_NAME \
    --policy-name $POLICY_NAME \
    --policy-document file://github-actions-deploy-role-policy.json

echo "✅ Policy applied successfully"

# Verify the policy was applied
echo "🔍 Verifying policy..."
if aws iam get-role-policy --role-name $ROLE_NAME --policy-name $POLICY_NAME >/dev/null 2>&1; then
    echo "✅ Policy verification successful"
else
    echo "❌ Policy verification failed"
    exit 1
fi

echo ""
echo "🎉 GitHub Actions Deploy Role permissions fixed!"
echo ""
echo "📝 What was fixed:"
echo "   • Added elasticloadbalancing:DescribeLoadBalancers permission"
echo "   • Added wafv2:GetWebACLForResource permission"
echo "   • Added ECS and ECR permissions for deployments"
echo "   • Added CloudWatch logs permissions"
echo ""
echo "🧪 The Security Smoke Test should now pass:"
echo "   • Workflow can describe the clarity-alb load balancer"
echo "   • Workflow can retrieve the associated WAF ACL"
echo "   • Workflow can verify WAF is named 'clarity-backend-rate-limiting'"
echo "   • Workflow can test SQL injection blocking"
echo ""
echo "🚀 Next steps:"
echo "   1. Push code to main branch to trigger the Security Smoke Test"
echo "   2. Monitor the workflow in GitHub Actions"
echo "   3. Verify all smoke tests pass" 