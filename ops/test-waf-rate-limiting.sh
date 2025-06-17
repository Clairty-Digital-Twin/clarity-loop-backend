#!/bin/bash
# AWS WAF Rate Limiting Test Script
# Tests that WAF rules are working correctly

set -e

# Load centralized configuration
source "$(dirname "$0")/env.sh"

# Configuration
TEST_ENDPOINT="/health"
RATE_LIMIT=100
HTTP_URL="http://$ALB_DNS"
HTTPS_URL="https://$ALB_DNS"

echo "🧪 Testing AWS WAF Rate Limiting for Clarity Digital Twin Backend"
echo "=================================================================="
echo "HTTP Target: $HTTP_URL$TEST_ENDPOINT"
echo "HTTPS Target: $HTTPS_URL$TEST_ENDPOINT" 
echo "Rate Limit: $RATE_LIMIT requests per 5 minutes"
echo "=================================================================="

# Test 1: Normal Request (should redirect to HTTPS)
echo ""
echo "TEST 1: Normal Request (should redirect to HTTPS)"
echo "-------------------------------------------------"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$ALB_URL$TEST_ENDPOINT")
if [ "$RESPONSE" = "301" ]; then
    echo "✅ HTTP→HTTPS redirect working: HTTP $RESPONSE (EXPECTED)"
elif [ "$RESPONSE" = "200" ]; then
    echo "✅ Direct response: HTTP $RESPONSE"
else
    echo "❌ Unexpected response: HTTP $RESPONSE"
fi

# Test 2: SQL Injection Attempt (should be blocked)
echo ""
echo "TEST 2: SQL Injection Attack (should be blocked)"
echo "------------------------------------------------"
SQL_INJECTION_URL="$ALB_URL$TEST_ENDPOINT?id=1%27%20OR%201=1--"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$SQL_INJECTION_URL")
if [ "$RESPONSE" = "403" ]; then
    echo "✅ SQL injection blocked: HTTP $RESPONSE (WAF working!)"
elif [ "$RESPONSE" = "200" ]; then
    echo "⚠️  SQL injection not blocked: HTTP $RESPONSE (WAF may not be active)"
else
    echo "❓ Unexpected response: HTTP $RESPONSE"
fi

# Test 3: Known Bad Input (should be blocked)
echo ""
echo "TEST 3: Known Bad Input Attack (should be blocked)"
echo "--------------------------------------------------"
BAD_INPUT_URL="$ALB_URL$TEST_ENDPOINT?payload=<script>alert('xss')</script>"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$BAD_INPUT_URL")
if [ "$RESPONSE" = "403" ]; then
    echo "✅ Bad input blocked: HTTP $RESPONSE (WAF working!)"
elif [ "$RESPONSE" = "200" ]; then
    echo "⚠️  Bad input not blocked: HTTP $RESPONSE (WAF may not be active)"
else
    echo "❓ Unexpected response: HTTP $RESPONSE"
fi

# Test 4: Rate Limiting Test (Burst Mode)
echo ""
echo "TEST 4: Rate Limiting Test (Burst of requests)"
echo "----------------------------------------------"
echo "Sending 10 rapid requests to test rate limiting..."

SUCCESS_COUNT=0
BLOCKED_COUNT=0

for i in {1..10}; do
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$ALB_URL$TEST_ENDPOINT")
    if [ "$RESPONSE" = "200" ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "Request $i: ✅ HTTP $RESPONSE"
    elif [ "$RESPONSE" = "429" ] || [ "$RESPONSE" = "403" ]; then
        BLOCKED_COUNT=$((BLOCKED_COUNT + 1))
        echo "Request $i: 🚫 HTTP $RESPONSE (Rate limited!)"
    else
        echo "Request $i: ❓ HTTP $RESPONSE"
    fi
    sleep 0.1  # Small delay between requests
done

echo ""
echo "Burst test results:"
echo "  Successful: $SUCCESS_COUNT"
echo "  Blocked: $BLOCKED_COUNT"

if [ $BLOCKED_COUNT -gt 0 ]; then
    echo "✅ Rate limiting appears to be working (some requests blocked)"
else
    echo "⚠️  No requests blocked in burst test"
fi

# Test 5: WAF Association Check
echo ""
echo "TEST 5: WAF Association Verification"
echo "------------------------------------"
if command -v aws &> /dev/null; then
    WAF_ASSOCIATION=$(aws wafv2 get-web-acl-for-resource \
        --region us-east-1 \
        --resource-arn $(aws elbv2 describe-load-balancers \
            --region us-east-1 \
            --names clarity-alb \
            --query 'LoadBalancers[0].LoadBalancerArn' \
            --output text) \
        --query 'WebACL.Name' \
        --output text 2>/dev/null || echo "None")
    
    if [ "$WAF_ASSOCIATION" != "None" ] && [ ! -z "$WAF_ASSOCIATION" ]; then
        echo "✅ WAF associated with ALB: $WAF_ASSOCIATION"
    else
        echo "❌ No WAF associated with ALB"
    fi
else
    echo "⚠️  AWS CLI not available - skipping association check"
fi

# Summary
echo ""
echo "🔒 WAF TEST SUMMARY"
echo "==================="
echo "ALB URL: $ALB_URL"
echo "Tests completed. Review results above."
echo ""
echo "💡 NOTES:"
echo "- Rate limiting may take time to trigger (5-minute windows)"
echo "- AWS managed rules may vary in blocking behavior"
echo "- Monitor CloudWatch metrics for detailed WAF activity"
echo ""
echo "📊 CloudWatch Metrics to Monitor:"
echo "   - clarity-rate-limit-blocked"
echo "   - clarity-common-rule-set"
echo "   - clarity-bad-inputs-blocked"
echo "   - clarity-sqli-blocked"
echo "   - clarity-ip-reputation-blocked" 