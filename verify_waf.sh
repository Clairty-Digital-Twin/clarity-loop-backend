#!/bin/bash
# Simple WAF verification script
set -e

echo "🔍 Verifying AWS WAF Deployment"
echo "==============================="

# Get ALB ARN
ALB_ARN=$(aws elbv2 describe-load-balancers \
    --names clarity-alb \
    --query 'LoadBalancers[0].LoadBalancerArn' \
    --output text \
    --region us-east-1)

echo "ALB ARN: $ALB_ARN"

# Check WAF association
echo ""
echo "Checking WAF association..."
aws wafv2 get-web-acl-for-resource \
    --resource-arn "$ALB_ARN" \
    --region us-east-1 > /tmp/waf_check.json

if grep -q "clarity-backend-rate-limiting" /tmp/waf_check.json; then
    echo "✅ WAF 'clarity-backend-rate-limiting' is associated with ALB"
    WAF_ID=$(grep -o '"Id":"[^"]*"' /tmp/waf_check.json | cut -d'"' -f4)
    echo "   WAF ID: $WAF_ID"
else
    echo "❌ No WAF associated or wrong WAF name"
fi

# Test endpoints
echo ""
echo "Testing protected endpoints..."

echo -n "SQL Injection test: "
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/health?id=1%27%20OR%201=1--")
if [ "$RESPONSE" = "403" ]; then
    echo "✅ BLOCKED (HTTP 403)"
else
    echo "❌ NOT BLOCKED (HTTP $RESPONSE)"
fi

echo -n "XSS test: "
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/health?payload=%3Cscript%3Ealert('xss')%3C/script%3E")
if [ "$RESPONSE" = "403" ]; then
    echo "✅ BLOCKED (HTTP 403)"
else
    echo "❌ NOT BLOCKED (HTTP $RESPONSE)"
fi

echo -n "Normal redirect: "
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/health")
if [ "$RESPONSE" = "301" ]; then
    echo "✅ HTTP→HTTPS REDIRECT (HTTP 301)"
else
    echo "❓ UNEXPECTED (HTTP $RESPONSE)"
fi

echo ""
echo "🔒 WAF VERIFICATION COMPLETE!"

# Cleanup
rm -f /tmp/waf_check.json 