#!/bin/bash

# Fix GitHub Actions OIDC Integration - 2025 Best Practices
# Based on official GitHub Actions and AWS documentation

set -e

ROLE_NAME="GitHubActionsDeploy"
OIDC_PROVIDER_URL="token.actions.githubusercontent.com"
ACCOUNT_ID="124355672559"
REGION="us-east-1"
GITHUB_ORG="Clarity-Digital-Twin"
GITHUB_REPO="clarity-loop-backend"

echo "🔧 Fixing GitHub Actions OIDC Integration (2025 Best Practices)"
echo "================================================================="
echo "📋 Role: $ROLE_NAME"
echo "📋 Account: $ACCOUNT_ID"
echo "📋 Region: $REGION"
echo "📋 Repository: $GITHUB_ORG/$GITHUB_REPO"
echo ""

# Step 1: Check if OIDC provider exists
echo "1️⃣ Checking OIDC Provider..."
if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "arn:aws:iam::$ACCOUNT_ID:oidc-provider/$OIDC_PROVIDER_URL" >/dev/null 2>&1; then
    echo "✅ OIDC Provider exists"
else
    echo "❌ OIDC Provider does not exist - creating it..."
    aws iam create-open-id-connect-provider \
        --url "https://$OIDC_PROVIDER_URL" \
        --client-id-list "sts.amazonaws.com" \
        --thumbprint-list "6938fd4d98bab03faadb97b34396831e3780aea1" \
        --thumbprint-list "1c58a3a8518e8759bf075b76b750d4f2df264fcd"
    echo "✅ OIDC Provider created"
fi

# Step 2: Update the trust policy with correct 2025 format
echo ""
echo "2️⃣ Updating Trust Policy with 2025 Format..."
aws iam update-assume-role-policy \
    --role-name $ROLE_NAME \
    --policy-document file://github-actions-trust-policy-2025.json

echo "✅ Trust policy updated with correct OIDC format"

# Step 3: Verify the trust policy
echo ""
echo "3️⃣ Verifying Trust Policy..."
TRUST_POLICY=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.AssumeRolePolicyDocument' --output json)
echo "$TRUST_POLICY" | jq '.'

# Step 4: Test the permissions
echo ""
echo "4️⃣ Testing Security Smoke Test Permissions..."
if aws elbv2 describe-load-balancers --names clarity-alb --region $REGION >/dev/null 2>&1; then
    echo "✅ ELB permissions working"
else
    echo "❌ ELB permissions failed"
fi

if aws wafv2 list-web-acls --scope REGIONAL --region $REGION >/dev/null 2>&1; then
    echo "✅ WAF permissions working"
else
    echo "❌ WAF permissions failed"
fi

echo ""
echo "🎉 OIDC Integration Fixed!"
echo ""
echo "📋 Next Steps:"
echo "   1. Push a commit to main branch"
echo "   2. Monitor GitHub Actions: https://github.com/$GITHUB_ORG/$GITHUB_REPO/actions"
echo "   3. Security Smoke Test should now pass"
echo ""
echo "🔗 Role ARN: arn:aws:iam::$ACCOUNT_ID:role/$ROLE_NAME" 