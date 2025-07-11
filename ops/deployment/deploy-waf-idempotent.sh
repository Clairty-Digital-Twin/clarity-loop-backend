#!/bin/bash
# AWS WAF Deployment Script - SINGULARITY EDITION
# Fully idempotent, production-ready, shock-the-world deployment
set -euo pipefail

# Load centralized configuration
source "$(dirname "$0")/env.sh"

echo "🚀 SINGULARITY-LEVEL AWS WAF DEPLOYMENT"
echo "======================================"
echo "🎯 Target: $ALB_NAME ($REGION)"
echo "🛡️  WAF: $WAF_NAME"
echo "🔄 Mode: IDEMPOTENT (Safe to re-run)"
echo "======================================"

# Step 1: Verify Prerequisites
echo "1️⃣  Verifying prerequisites..."
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ AWS CLI not configured"
    exit 1
fi

if [ ! -f "ops/aws-waf-rate-limiting.json" ]; then
    echo "❌ WAF config file missing"
    exit 1
fi

echo "✅ Prerequisites verified"

# Step 2: Check if WAF already exists
echo "2️⃣  Checking existing WAF deployment..."
EXISTING_WAF_ARN=$(aws wafv2 list-web-acls \
    --region "$REGION" \
    --scope REGIONAL \
    --query "WebACLs[?Name=='$WAF_NAME'].ARN | [0]" \
    --output text 2>/dev/null || echo "None")

if [ "$EXISTING_WAF_ARN" != "None" ] && [ "$EXISTING_WAF_ARN" != "null" ]; then
    echo "✅ WAF already exists: $EXISTING_WAF_ARN"
    WAF_ARN="$EXISTING_WAF_ARN"
    WAF_ID=$(echo "$WAF_ARN" | sed 's/.*\///')
else
    # Step 3: Create WAF Web ACL
    echo "3️⃣  Creating new WAF Web ACL..."
    WAF_RESULT=$(aws wafv2 create-web-acl \
        --region "$REGION" \
        --cli-input-json file://ops/aws-waf-rate-limiting.json)
    
    WAF_ID=$(echo "$WAF_RESULT" | jq -r '.Summary.Id')
    WAF_ARN=$(echo "$WAF_RESULT" | jq -r '.Summary.ARN')
    
    echo "✅ Created WAF: $WAF_ID"
fi

# Step 4: Check if already associated
echo "4️⃣  Checking WAF association..."
CURRENT_ASSOCIATION=$(aws wafv2 get-web-acl-for-resource \
    --region "$REGION" \
    --resource-arn "$ALB_ARN" \
    --query 'WebACL.Id' \
    --output text 2>/dev/null || echo "None")

if [ "$CURRENT_ASSOCIATION" = "$WAF_ID" ]; then
    echo "✅ WAF already associated with ALB"
else
    # Step 5: Associate WAF with ALB
    echo "5️⃣  Associating WAF with ALB..."
    aws wafv2 associate-web-acl \
        --region "$REGION" \
        --web-acl-arn "$WAF_ARN" \
        --resource-arn "$ALB_ARN"
    echo "✅ WAF associated with ALB"
fi

# Step 6: Ensure WAF logging is enabled
echo "6️⃣  Configuring WAF logging..."
LOG_GROUP="aws-waf-logs-clarity-backend"

# Create log group if it doesn't exist
aws logs create-log-group --log-group-name "$LOG_GROUP" --region "$REGION" 2>/dev/null || true

# Check if logging is already configured
LOGGING_STATUS=$(aws wafv2 get-logging-configuration \
    --resource-arn "$WAF_ARN" \
    --region "$REGION" \
    --query 'LoggingConfiguration.ResourceArn' \
    --output text 2>/dev/null || echo "None")

if [ "$LOGGING_STATUS" = "None" ]; then
    aws wafv2 put-logging-configuration \
        --logging-configuration file://ops/waf-logging-config.json \
        --region "$REGION"
    echo "✅ WAF logging enabled"
else
    echo "✅ WAF logging already configured"
fi

# Step 7: Final verification
echo "7️⃣  Final verification..."
FINAL_CHECK=$(aws wafv2 get-web-acl-for-resource \
    --region "$REGION" \
    --resource-arn "$ALB_ARN" \
    --query 'WebACL.Name' \
    --output text)

if [ "$FINAL_CHECK" = "$WAF_NAME" ]; then
    echo "✅ DEPLOYMENT VERIFIED"
else
    echo "❌ VERIFICATION FAILED"
    exit 1
fi

# Success banner
echo ""
echo "🎉 SINGULARITY-LEVEL WAF DEPLOYMENT COMPLETE!"
echo "=============================================="
echo "🛡️  WAF ID: $WAF_ID"
echo "🔗 ALB: $ALB_NAME"
echo "📊 Logging: CloudWatch ($LOG_GROUP)"
echo "🔄 Status: IDEMPOTENT (Safe to re-run)"
echo ""
echo "🚨 PROTECTION ACTIVE:"
echo "   ⚡ Rate Limiting: 100 req/5min per IP"
echo "   🛡️  OWASP Top 10: SQL injection, XSS, etc."
echo "   🚫 Bad Input Blocking: Malicious payloads"
echo "   📡 IP Reputation: Known bad actors"
echo ""
echo "🎯 READY TO SHOCK THE TECH WORLD! 🚀" 