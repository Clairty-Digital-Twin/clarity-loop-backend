#!/bin/bash
# AWS WAF Rate Limiting Deployment Script
# Implements Task 3: AWS WAF Rate Limiting for DDoS Protection

set -e  # Exit on any error

# Load centralized configuration
source "$(dirname "$0")/env.sh"

# Configuration
WAF_CONFIG_FILE="ops/aws-waf-rate-limiting.json"

echo "🔒 Deploying AWS WAF Rate Limiting for Clarity Digital Twin Backend"
echo "=================================="
echo "Region: $REGION"
echo "ALB: $ALB_NAME"
echo "WAF Name: $WAF_NAME"
echo "=================================="

# Step 1: Verify AWS CLI is configured
echo "1. Verifying AWS CLI configuration..."
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ ERROR: AWS CLI not configured or no valid credentials"
    echo "Run: aws configure"
    exit 1
fi
echo "✅ AWS CLI configured"

# Step 2: Get ALB ARN
echo "2. Finding Application Load Balancer ARN..."
# ALB_ARN already loaded from env.sh

if [ "$ALB_ARN" = "None" ] || [ -z "$ALB_ARN" ]; then
    echo "❌ ERROR: Could not find ALB with name: $ALB_NAME"
    echo "Available load balancers:"
    aws elbv2 describe-load-balancers --region $AWS_REGION --query 'LoadBalancers[].LoadBalancerName'
    exit 1
fi
echo "✅ Found ALB: $ALB_ARN"

# Step 3: Get or create WAF Web ACL (IDEMPOTENT)
echo "3. Checking for existing WAF Web ACL..."
WAF_ARN=$(aws wafv2 list-web-acls \
    --region $REGION \
    --scope REGIONAL \
    --query "WebACLs[?Name=='$WAF_NAME'].ARN | [0]" \
    --output text)

if [ "$WAF_ARN" = "None" ] || [ -z "$WAF_ARN" ]; then
    echo "4. Creating new WAF Web ACL..."
    WAF_RESULT=$(aws wafv2 create-web-acl \
        --region $AWS_REGION \
        --cli-input-json file://$WAF_CONFIG_FILE)
    
    WAF_ID=$(echo $WAF_RESULT | jq -r '.Summary.Id')
    WAF_ARN=$(echo $WAF_RESULT | jq -r '.Summary.ARN')
    echo "✅ Created WAF: $WAF_ID"
else
    echo "✅ WAF already exists: $WAF_ARN"
    WAF_ID=$(echo "$WAF_ARN" | sed 's/.*\///')
fi

if [ -z "$WAF_ID" ] || [ "$WAF_ID" = "null" ]; then
    echo "❌ ERROR: Failed to create WAF Web ACL"
    echo "Response: $WAF_RESULT"
    exit 1
fi
echo "✅ Created WAF Web ACL: $WAF_ID"
echo "   ARN: $WAF_ARN"

# Step 5: Associate WAF with ALB
echo "5. Associating WAF Web ACL with Application Load Balancer..."
aws wafv2 associate-web-acl \
    --region $AWS_REGION \
    --web-acl-arn $WAF_ARN \
    --resource-arn $ALB_ARN

echo "✅ Associated WAF with ALB"

# Step 6: Verify association
echo "6. Verifying WAF association..."
ASSOCIATED_WAF=$(aws wafv2 get-web-acl-for-resource \
    --region $AWS_REGION \
    --resource-arn $ALB_ARN \
    --query 'WebACL.Id' \
    --output text)

if [ "$ASSOCIATED_WAF" = "$WAF_ID" ]; then
    echo "✅ WAF successfully associated with ALB"
else
    echo "❌ ERROR: WAF association verification failed"
    exit 1
fi

# Step 7: Display WAF rules summary
echo ""
echo "🔒 AWS WAF DEPLOYMENT COMPLETE!"
echo "=================================="
echo "WAF Web ACL ID: $WAF_ID"
echo "WAF Web ACL ARN: $WAF_ARN"
echo "Associated with ALB: $ALB_NAME"
echo ""
echo "🛡️  ACTIVE PROTECTION RULES:"
echo "1. ⚡ Rate Limiting: 100 requests per 5 minutes per IP"
echo "2. 🛡️  Common Rule Set: OWASP Top 10 protection"
echo "3. 🚫 Known Bad Inputs: Malicious payload blocking"  
echo "4. 🔍 SQL Injection: SQL attack protection"
echo "5. 🚨 IP Reputation: Malicious IP blocking"
echo ""
echo "📊 CloudWatch Metrics:"
echo "   - clarity-rate-limit-blocked"
echo "   - clarity-common-rule-set"
echo "   - clarity-bad-inputs-blocked"
echo "   - clarity-sqli-blocked"
echo "   - clarity-ip-reputation-blocked"
echo ""
echo "🧪 TESTING:"
echo "   Rate limit test: curl -s http://$ALB_NAME.elb.amazonaws.com/health (repeat 101+ times in 5min)"
echo "   SQL injection test: curl 'http://$ALB_NAME.elb.amazonaws.com/?id=1%27%20OR%201=1--'"
echo ""
echo "💰 ESTIMATED COST: ~$5-10/month"
echo "🔒 SECURITY LEVEL: HIGH - DDoS and attack protection active"
echo ""
echo "✅ TASK 3: AWS WAF RATE LIMITING COMPLETE!" 