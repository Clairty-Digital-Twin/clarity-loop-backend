#!/usr/bin/env bash
set -euo pipefail

# Cognito Security Analysis Script
USER_POOL_ID="us-east-1_efXaR5EcP"
REGION="us-east-1"

echo "🔍 ANALYZING COGNITO USER POOL SECURITY SETTINGS"
echo "User Pool ID: $USER_POOL_ID"
echo "Region: $REGION"
echo ""

echo "📋 BASIC POOL INFO:"
aws cognito-idp describe-user-pool \
    --user-pool-id "$USER_POOL_ID" \
    --region "$REGION" \
    --query 'UserPool.{Id:Id,Name:Name,Status:Status,CreationDate:CreationDate}' \
    --output table

echo ""
echo "🔐 PASSWORD POLICIES:"
aws cognito-idp describe-user-pool \
    --user-pool-id "$USER_POOL_ID" \
    --region "$REGION" \
    --query 'UserPool.Policies.PasswordPolicy' \
    --output table

echo ""
echo "🛡️ MFA CONFIGURATION:"
aws cognito-idp describe-user-pool \
    --user-pool-id "$USER_POOL_ID" \
    --region "$REGION" \
    --query 'UserPool.{MfaConfiguration:MfaConfiguration,MfaTypes:MfaTypes}' \
    --output table

echo ""
echo "🚨 ADVANCED SECURITY FEATURES:"
aws cognito-idp describe-user-pool \
    --user-pool-id "$USER_POOL_ID" \
    --region "$REGION" \
    --query 'UserPool.UserPoolAddOns' \
    --output table

echo ""
echo "📧 VERIFICATION & NOTIFICATIONS:"
aws cognito-idp describe-user-pool \
    --user-pool-id "$USER_POOL_ID" \
    --region "$REGION" \
    --query 'UserPool.{AutoVerifiedAttributes:AutoVerifiedAttributes,AccountRecoverySetting:AccountRecoverySetting}' \
    --output table

echo ""
echo "🎯 SIGN-UP CONFIGURATION:"
aws cognito-idp describe-user-pool \
    --user-pool-id "$USER_POOL_ID" \
    --region "$REGION" \
    --query 'UserPool.{AdminCreateUserConfig:AdminCreateUserConfig}' \
    --output table

echo ""
echo "✅ ANALYSIS COMPLETE - Ready for lockout policy configuration!"