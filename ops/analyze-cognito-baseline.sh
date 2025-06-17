#!/usr/bin/env bash
set -euo pipefail

# Cognito Security Baseline Analysis
USER_POOL_ID="us-east-1_efXaR5EcP"
REGION="us-east-1"

echo "🔍 COGNITO SECURITY BASELINE ANALYSIS"
echo "======================================"
echo "User Pool ID: $USER_POOL_ID"
echo "Region: $REGION"
echo ""

echo "📋 BASIC POOL INFO:"
aws cognito-idp describe-user-pool \
    --user-pool-id "$USER_POOL_ID" \
    --region "$REGION" \
    --query 'UserPool.{Name:Name,Status:Status,MfaConfiguration:MfaConfiguration}' \
    --output table

echo ""
echo "🔐 PASSWORD POLICIES:"
aws cognito-idp describe-user-pool \
    --user-pool-id "$USER_POOL_ID" \
    --region "$REGION" \
    --query 'UserPool.Policies.PasswordPolicy' \
    --output table

echo ""
echo "🛡️ ACCOUNT TAKEOVER PROTECTION:"
aws cognito-idp describe-user-pool \
    --user-pool-id "$USER_POOL_ID" \
    --region "$REGION" \
    --query 'UserPool.UserPoolAddOns' \
    --output table

echo ""
echo "⚙️ ADVANCED SECURITY FEATURES:"
aws cognito-idp describe-user-pool \
    --user-pool-id "$USER_POOL_ID" \
    --region "$REGION" \
    --query 'UserPool.{AutoVerifiedAttributes:AutoVerifiedAttributes,AliasAttributes:AliasAttributes,UsernameAttributes:UsernameAttributes}' \
    --output table

echo ""
echo "🚨 CURRENT LOCKOUT STATUS:"
echo "Checking if advanced security features are enabled..."

# Check if advanced security is enabled
ADVANCED_SECURITY=$(aws cognito-idp describe-user-pool \
    --user-pool-id "$USER_POOL_ID" \
    --region "$REGION" \
    --query 'UserPool.UserPoolAddOns.AdvancedSecurityMode' \
    --output text)

if [ "$ADVANCED_SECURITY" = "ENFORCED" ]; then
    echo "✅ Advanced Security: ENFORCED (Account takeover protection active)"
elif [ "$ADVANCED_SECURITY" = "AUDIT" ]; then
    echo "⚠️  Advanced Security: AUDIT ONLY (Monitoring but not blocking)"
else
    echo "❌ Advanced Security: DISABLED (No account takeover protection)"
fi

echo ""
echo "🎯 BASELINE ANALYSIS COMPLETE!"
echo "Check the output above to determine current security posture."